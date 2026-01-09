from mlops.model import Model
from mlops.data import MyDataset
import typer
import torch

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10):
    dataset = MyDataset("data/raw")
    model = Model()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

if __name__ == "__main__":
    typer.run(train)
