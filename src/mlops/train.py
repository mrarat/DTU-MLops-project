from mlops.model import Model
from mlops.data import load_data
import torch
import os
import wandb
from google.cloud import storage
import typer


def upload_to_gcs(local_file, bucket, gcs_path):
    client = storage.Client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_file)
    print(f"Uploaded {local_file} to gs://{bucket}/{gcs_path}")

def download_from_gcs(bucket, gcs_path, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(gcs_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def train():
    batch_size = 32
    epochs = 10
    lr = 1e-3
    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Logging
    run = wandb.init(project='playing-cards-mlops',
               job_type="train",
               config={
                   'epochs':epochs,
                   'batch size':batch_size,
                   'learning rate':lr,'seed':seed})

    # Loading data
    download_from_gcs("dtu-mlops-group-48-data","data/processed/train.pt","data/processed/train.pt")
    train_set = load_data(split = "train")
    model = Model().to(DEVICE)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn =  torch.nn.CrossEntropyLoss(ignore_index=-1) # Using Cross entropy, ignore labels of -1 (missing)
    rank_weight = 1
    suit_weight = 1 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    statistics = {"train_loss": [], "train_accuracy_suit": [], "train_accuracy_rank": []}

    # start training
    for epoch in range(epochs):
        model.train()
        for i, (img, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            img = (img.float() / 255.0).to(DEVICE)  # convert to float in [0,1]

            targets = targets.to(DEVICE, dtype=torch.long)
            rank_targets = targets[:, 0]
            suit_targets = targets[:, 1]

            #predict
            y_pred = model(img)
            
            # compute loss
            loss = suit_weight*loss_fn(y_pred['suit'], suit_targets) + rank_weight*loss_fn(y_pred['rank'], rank_targets)  # calculating loss as sum of the seperate losses
            # gradient step
            loss.backward()
            optimizer.step()

            # Statistics
            statistics["train_loss"].append(loss.item())

            r_accuracy = (y_pred['rank'].argmax(dim=1) == rank_targets).float().mean().item()
            s_accuracy = (y_pred['suit'].argmax(dim=1) == suit_targets).float().mean().item()

            statistics["train_accuracy_rank"].append(r_accuracy)
            statistics["train_accuracy_suit"].append(s_accuracy)

            # Logging
            wandb.log({'loss':loss.item(),'rank accuracy':r_accuracy,'suit accuracy':s_accuracy})
            

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, "
                      f"rank_acc: {r_accuracy:.4f}, suit_acc: {s_accuracy:.4f}")

    print("Training complete")   

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")
    upload_to_gcs("models/model.pth", "dtu-mlops-group-48-data", "models/model.pth")

    # Log as W&B artifact
    model_artifact = wandb.Artifact(
        name="card-deck_model",
        type="model",
    )
    model_artifact.add_file("models/model.pth")
    run.log_artifact(model_artifact)

    wandb.finish()

if __name__ == "__main__":
    typer.run(train)