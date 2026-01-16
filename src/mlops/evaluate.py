from mlops.model import Model
from mlops.data import load_data

import torch
import typer
import wandb
from google.cloud import storage
from io import BytesIO
import os

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

BUCKET_NAME = "dtu-mlops-group-48-data"
MODEL_FILE = "models/model.pth"


def load_model_from_gcs(bucket_name: str, model_file: str, device: torch.device):
    """Load a PyTorch model checkpoint directly from Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_file)
    
    if not blob.exists():
        raise FileNotFoundError(f"{model_file} not found in bucket {bucket_name}")

    checkpoint_bytes = blob.download_as_bytes()
    checkpoint = torch.load(BytesIO(checkpoint_bytes), map_location=device)

    model = Model().to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def download_from_gcs(bucket, gcs_path, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(gcs_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)


def evaluate(batch_size: int = 32) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(f"Evaluating model from bucket {BUCKET_NAME}/{MODEL_FILE}...")
    
    # WandB init
    run = wandb.init(
        project="playing-cards-mlops",
        job_type="evaluation",
        config={
            "checkpoint": MODEL_FILE,
            "batch_size": batch_size,
            "device": str(DEVICE),
        },
    )
    # Load model directly from GCS
    model = load_model_from_gcs(BUCKET_NAME, MODEL_FILE, DEVICE)

    download_from_gcs("dtu-mlops-group-48-data","data/processed/test.pt","data/processed/test.pt")

    test_set = load_data(split = "test")
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    model.eval()

    rank_correct = 0
    suit_correct = 0
    both_correct = 0
    n = 0
    
    with torch.no_grad():
        for img, targets in test_dataloader:
            img = (img.float() / 255.0).to(DEVICE) # convert to float in [0,1]
            targets = targets.to(DEVICE, dtype=torch.long)
            rank_targets = targets[:, 0]
            suit_targets = targets[:, 1]

            out = model(img)
            rank_pred = out["rank"].argmax(dim=1)
            suit_pred = out["suit"].argmax(dim=1)
            
            rank_correct += (rank_pred == rank_targets).sum().item()
            suit_correct += (suit_pred == suit_targets).sum().item()
            both_correct += ((rank_pred == rank_targets) & (suit_pred == suit_targets)).sum().item()
            n += img.size(0)
            
    rank_acc  = rank_correct / n
    suit_acc  = suit_correct / n
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
    wandb.finish()

if __name__ == "__main__":
    typer.run(evaluate)