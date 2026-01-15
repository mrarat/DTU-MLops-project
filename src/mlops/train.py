from mlops.model import Model
from mlops.data import load_data
import torch
import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# @hydra.main(version_base = None, config_path = "configs", config_name = "config.yaml")
# def train(cfg):
def train():
    # batch_size = cfg.hyperparameters.batch_size
    # epochs = cfg.hyperparameters.epochs
    # lr = cfg.hyperparameters.lr
    # seed = cfg.hyperparameters.seed

    batch_size = 32
    epochs = 10
    lr = 1e-3
    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Logging
    run = wandb.init(
        project="playing-cards-mlops",
        job_type="train",
        config={"epochs": epochs, "batch size": batch_size, "learning rate": lr, "seed": seed},
    )

    # Loading data
    train_set = load_data(split="train")
    wandb.init(
        project="playing-cards-mlops",
        config={"epochs": epochs, "batch size": batch_size, "learning rate": lr, "seed": seed},
    )

    # Loading data
    train_set = load_data(split="train")
    test_set = load_data(split="test")
    model = Model().to(DEVICE)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Using Cross entropy, ignore labels of -1 (missing)
    rank_weight = 1
    suit_weight = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy_suit": [], "train_accuracy_rank": []}

    # start training
    for epoch in range(epochs):
        model.train()
        for i, (img, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            img = (img.float() / 255.0).to(DEVICE)
            target = target.to(DEVICE)
            rank_t = target[:, 0]  # 0 is rank
            suit_t = target[:, 1]  # 1 is suit
            img = img.float() / 255.0  # convert to float in [0,1]
            img = img.to(DEVICE)
            # print(target)
            target[:, 0] = target[:, 0].to(DEVICE)  # 0 is rank
            target[:, 1] = target[:, 1].to(DEVICE)  # 1 is suit

            # predict
            y_pred = model(img)

            # compute loss
            loss = suit_weight * loss_fn(y_pred["suit"], suit_t) + rank_weight * loss_fn(y_pred["rank"], rank_t)
            loss = suit_weight * loss_fn(y_pred["suit"], target[:, 1]) + rank_weight * loss_fn(
                y_pred["rank"], target[:, 0]
            )  # calculating loss as sum of the seperate losses
            # gradient step
            loss.backward()
            optimizer.step()

            # Statistics
            statistics["train_loss"].append(loss.item())

            r_accuracy = (y_pred["rank"].argmax(dim=1) == rank_targets).float().mean().item()
            s_accuracy = (y_pred["suit"].argmax(dim=1) == suit_targets).float().mean().item()
            r_accuracy = (y_pred["rank"].argmax(dim=1) == target[:, 0]).float().mean().item()
            s_accuracy = (y_pred["suit"].argmax(dim=1) == target[:, 1]).float().mean().item()

            statistics["train_accuracy_rank"].append(r_accuracy)
            statistics["train_accuracy_suit"].append(s_accuracy)

            # Logging

            if i % 100 == 0:
                print(
                    f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, "
                    f"rank_acc: {r_accuracy:.4f}, suit_acc: {s_accuracy:.4f}"
                )
                wandb.log({"loss": loss.item(), "rank accuracy": r_accuracy, "suit accuracy": s_accuracy})
            wandb.log({"loss": loss.item(), "rank accuracy": r_accuracy, "suit accuracy": s_accuracy})

    print("Training complete")

    # Save model
    # save_dir = os.path.join(get_original_cwd(), "models")
    torch.save(model.state_dict(), "models/model.pth")
    # torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

    # Log as W&B artifact
    model_artifact = wandb.Artifact(
        name="card-deck_model",
        type="model",
    )
    model_artifact.add_file("models/model.pth")
    run.log_artifact(model_artifact)


if __name__ == "__main__":
    train()
