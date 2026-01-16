from mlops.model import Model
from mlops.data import load_data
import torch
import hydra
import os
import wandb
import random
import numpy as np
from hydra.utils import get_original_cwd
from google.cloud import storage
from google.oauth2 import service_account
import typer
from typing import Any, Dict
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pathlib import Path


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def upload_to_gcs(local_file, bucket, gcs_path) -> None:
    credentials_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = storage.Client(credentials=credentials, project="dtu-mlops-group-48")
    bucket = client.bucket(bucket)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_file)
    print(f"Uploaded {local_file} to gs://{bucket}/{gcs_path}")


def download_from_gcs(bucket, gcs_path, local_path):
    print("Downloading from GCS...")
    credentials_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    client = storage.Client(credentials=credentials, project="dtu-mlops-group-48")
    bucket = client.bucket(bucket)
    blob = bucket.blob(gcs_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)


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
    
    models_dir = os.path.join(get_original_cwd(), "models")
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    download_from_gcs("dtu-mlops-group-48-data","data/processed/train.pt","data/processed/train.pt")
    download_from_gcs("dtu-mlops-group-48-data","data/processed/valid.pt","data/processed/valid.pt")

    train_set = load_data(processed_dir=os.path.join(get_original_cwd(), "data/processed"), split="train")
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_set = load_data(processed_dir=os.path.join(get_original_cwd(), "data/processed"), split="valid")
    eval_dataloader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False)


    model = Model().to(DEVICE)
    loss_fn =  torch.nn.CrossEntropyLoss()
    
    r_weight = 0.5
    s_weight = 1 - r_weight
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_avg_acc = 0.0
    best_model_path = None
    
    step = 0
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        rank_correct = suit_correct = card_correct = 0
        n = 0
        for i, (img, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            img = (img.float() / 255.0).to(DEVICE)
            targets = targets.to(DEVICE)
            
            rank_t = targets[:, 0]
            suit_t = targets[:, 1]

            # Predict
            y_pred = model(img)
            loss = s_weight*loss_fn(y_pred['suit'], suit_t) + r_weight*loss_fn(y_pred['rank'], rank_t)
            
            
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * img.size(0)
            
            rank_correct += (y_pred["rank"].argmax(1) == rank_t).sum().item()
            suit_correct += (y_pred["suit"].argmax(1) == suit_t).sum().item()
            card_correct += ((y_pred["rank"].argmax(1) == rank_t) & (y_pred["suit"].argmax(1) == suit_t)).sum().item()
            n += img.size(0)
            step += 1


        # Compute accuracy
   
        train_loss = train_loss_sum / n
        train_r_acc = rank_correct / n
        train_s_acc = suit_correct / n
        train_card_acc = card_correct / n
        train_avg_acc = (train_r_acc + train_s_acc) / 2

        
            
        model.eval()
        rank_correct = 0
        suit_correct = 0
        card_correct = 0
        eval_loss_sum = 0.0
        n = 0
        with torch.no_grad():
            for i, (img, targets) in enumerate(eval_dataloader):
                img = (img.float() / 255.0).to(DEVICE)
                targets = targets.to(DEVICE)
                
                rank_t = targets[:, 0]
                suit_t = targets[:, 1]
                
                # Predict
                y_pred = model(img)
                eval_loss = s_weight*loss_fn(y_pred['suit'], suit_t) + r_weight*loss_fn(y_pred['rank'], rank_t)
                eval_loss_sum += eval_loss.item() * img.size(0)
                
                
                rank_correct += (y_pred["rank"].argmax(dim=1) == rank_t).sum().item()
                suit_correct += (y_pred["suit"].argmax(dim=1) == suit_t).sum().item()
                card_correct += ((y_pred["rank"].argmax(dim=1) == rank_t) & (y_pred["suit"].argmax(dim=1) == suit_t)).sum().item()
                n += img.size(0)

        val_r_acc = rank_correct / n
        val_s_acc = suit_correct / n
        val_card_acc = card_correct / n
        val_avg_acc = (val_r_acc + val_s_acc) / 2
        val_loss = eval_loss_sum / n
        
        if val_avg_acc > (best_val_avg_acc + 1e-4):
            best_val_avg_acc = val_avg_acc
            best_model_path = os.path.join(get_original_cwd(), "models", "best_model.pth")
            torch.save(model.state_dict(), best_model_path)

        
        wandb.log(
            {
                "epoch:": epoch,
                "train/loss": train_loss,
                "train/rank_accuracy": train_r_acc,
                "train/suit_accuracy": train_s_acc,
                "train/card_accuracy": train_card_acc,
                "train/avg_accuracy": train_avg_acc,
                
                "eval/val_loss": val_loss,
                "eval/rank_accuracy": val_r_acc,
                "eval/suit_accuracy": val_s_acc,
                "eval/card_accuracy": val_card_acc,
                "eval/avg_accuracy": val_avg_acc,
                "eval/best_avg_accuracy": best_val_avg_acc, 
            },
            step=step
        )

    print("Training complete")
    
    if best_model_path is not None:
        model_artifact = wandb.Artifact(
            name="card-deck_best_model",
            type="model",
            metadata={"eval/best_avg_accuracy": best_val_avg_acc},
        )
        model_artifact.add_file(best_model_path)
        wandb.log_artifact(model_artifact)
        upload_to_gcs(best_model_path, "dtu-mlops-group-48-data", "models/best_model.pth")

    wandb.finish()
    
@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="defaults")
def main(cfg: DictConfig) -> None:
    train(cfg)

if __name__ == "__main__":
    main()