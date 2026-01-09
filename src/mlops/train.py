from mlops.model import Model
from mlops.data import fetch_cards
import torch
import hydra
import os
from hydra.utils import get_original_cwd

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

@hydra.main(version_base = None, config_path = "configs", config_name = "config.yaml")
def train(cfg):
    batch_size = cfg.hyperparameters.batch_size
    epochs = cfg.hyperparameters.epochs
    lr = cfg.hyperparameters.lr
    seed = cfg.hyperparameters.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_set, test_set = fetch_cards() # FIXME:missing how to correctly load data 
    model = Model().to(DEVICE)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss() # Using Cross entropy 
    rank_weight = 1
    suit_weight = 1 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy_suit": [], "train_accuracy_rank": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            optimizer.zero_grad()

            img = img.to(DEVICE)
            target['rank'] = target['rank'].to(DEVICE)
            target['suit'] = target['suit'].to(DEVICE)
            y_pred = model(img)
            
            loss = suit_weight*loss_fn(y_pred['suit'], target['suit']) + rank_weight*loss_fn(y_pred['rank'], target['rank'])  # calculating loss as sum of the seperate losses
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            r_accuracy = (y_pred['rank'].argmax(dim=1) == target['rank']).float().mean().item()
            s_accuracy = (y_pred['suit'].argmax(dim=1) == target['suit']).float().mean().item()

            statistics["train_accuracy_rank"].append(r_accuracy)
            statistics["train_accuracy_suit"].append(s_accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, "
                      f"rank_acc: {r_accuracy:.4f}, suit_acc: {s_accuracy:.4f}")

    print("Training complete")   
    save_dir = os.path.join(get_original_cwd(), "models")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

if __name__ == "__main__":
    train()
