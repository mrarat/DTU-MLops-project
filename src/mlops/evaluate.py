from mlops.model import Model
from mlops.data import load_data

import torch
import typer
import wandb
# from pathlib import Path
# import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str, batch_size: int = 32) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    # WandB init
    wandb.init(
        project="playing-cards-mlops",
        job_type="evaluation",
        config={
            "checkpoint": model_checkpoint,
            "batch_size": batch_size,
            "device": str(DEVICE),
        },
    )

    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    test_set = load_data(split="test")
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    model.eval()
    rank_correct = 0
    suit_correct = 0
    both_correct = 0
    n = 0

    with torch.no_grad():
        for img, targets in test_dataloader:
            img = (img.float() / 255.0).to(DEVICE)  # convert to float in [0,1]
            targets = targets.to(DEVICE, dtype=torch.long)
            rank_targets = targets[:, 0]
            suit_targets = targets[:, 1]
        for img, rank_targets, suit_targets in test_dataloader:
            img = img.to(DEVICE)
            rank_targets = rank_targets.to(DEVICE)
            suit_targets = suit_targets.to(DEVICE)

            out = model(img)
            rank_pred = out["rank"].argmax(dim=1)
            suit_pred = out["suit"].argmax(dim=1)

            rank_correct += (rank_pred == rank_targets).sum().item()
            suit_correct += (suit_pred == suit_targets).sum().item()
            both_correct += ((rank_pred == rank_targets) & (suit_pred == suit_targets)).sum().item()
            n += img.size(0)

    rank_acc = rank_correct / n
    suit_acc = suit_correct / n
    joint_acc = both_correct / n

    # Log in wandB
    wandb.log(
        {
            "eval/rank_accuracy": rank_acc,
            "eval/suit_accuracy": suit_acc,
            "eval/joint_accuracy": joint_acc,
            "eval/num_samples": n,
        }
    )
    print(f"rank_accuracy {rank_acc}, eval/suit_accuracy {suit_acc}, joint_accuracy {joint_acc}, num_samples {n}")


if __name__ == "__main__":
    typer.run(evaluate)
