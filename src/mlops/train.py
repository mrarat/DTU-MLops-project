from typing import Any, Dict
from mlops.model import Model
from mlops.data import load_data
import torch
import hydra
import os
from hydra.utils import get_original_cwd
import wandb
import random
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def train(cfg: DictConfig) -> None:
    # Resolve interpolations + convert to plain Python for W&B
    resolved_cfg: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)

    # Initialize WandB
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, job_type="train", config=resolved_cfg)

    batch_size = cfg.hyperparameters.batch_size
    epochs = cfg.hyperparameters.epochs
    lr = cfg.hyperparameters.lr
    seed = cfg.hyperparameters.seed

    set_seed(seed)

    # Loading data
    train_set = load_data(processed_dir=os.path.join(get_original_cwd(), "data/processed"), split="train")
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_set = load_data(processed_dir=os.path.join(get_original_cwd(), "data/processed"), split="valid")
    eval_dataloader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    model = Model().to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()

    r_weight = 0.5
    s_weight = 1 - r_weight
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    step = 0
    for epoch in range(epochs):
        model.train()
        for i, (img, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            img = (img.float() / 255.0).to(DEVICE)
            targets = targets.to(DEVICE)

            rank_t = targets[:, 0]
            suit_t = targets[:, 1]

            y_pred = model(img)
            loss = s_weight * loss_fn(y_pred["suit"], suit_t) + r_weight * loss_fn(y_pred["rank"], rank_t)

            # gradient step
            loss.backward()
            optimizer.step()

            # Statistics
            # statistics["train_loss"].append(loss.item())

            r_acc = (y_pred["rank"].argmax(dim=1) == rank_t).float().mean().item()
            s_acc = (y_pred["suit"].argmax(dim=1) == suit_t).float().mean().item()
            card_acc = (
                ((y_pred["rank"].argmax(dim=1) == rank_t) & (y_pred["suit"].argmax(dim=1) == suit_t))
                .float()
                .mean()
                .item()
            )
            avg_acc = (r_acc + s_acc) / 2

            # Logging
            if step % 20 == 0:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "rank_accuracy": r_acc,
                        "suit_accuracy": s_acc,
                        "card_accuracy": card_acc,
                        "avg_accuracy": avg_acc,
                        "epoch": epoch,
                    },
                    step=step,
                )
            if i % 100 == 0:
                print(
                    f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, "
                    f"rank_acc: {r_acc:.4f}, suit_acc: {s_acc:.4f}"
                )

            step += 1

        model.eval()
        rank_correct = 0
        suit_correct = 0
        both_correct = 0
        n = 0
        with torch.no_grad():
            for i, (img, targets) in enumerate(eval_dataloader):
                img = (img.float() / 255.0).to(DEVICE)
                targets = targets.to(DEVICE)

                rank_targets = targets[:, 0]
                suit_targets = targets[:, 1]

                out = model(img)
                rank_pred = out["rank"].argmax(dim=1)
                suit_pred = out["suit"].argmax(dim=1)

                rank_correct += (rank_pred == rank_targets).sum().item()
                suit_correct += (suit_pred == suit_targets).sum().item()
                both_correct += ((rank_pred == rank_targets) & (suit_pred == suit_targets)).sum().item()
                n += img.size(0)

        val_r_acc = rank_correct / n
        val_s_acc = suit_correct / n
        val_card_acc = both_correct / n
        val_avg_acc = (val_r_acc + val_s_acc) / 2

        # Log in wandB
        wandb.log(
            {
                "eval/rank_accuracy": val_r_acc,
                "eval/suit_accuracy": val_s_acc,
                "eval/card_accuracy": val_card_acc,
                "eval/avg_accuracy": val_avg_acc,
                "eval/num_samples": n,
            }
        )

    print("Training complete")

    save_dir = os.path.join(get_original_cwd(), "models")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    model_artifact = wandb.Artifact(name="card-deck_model", type="model")
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)

    wandb.finish()


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="defaults")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
