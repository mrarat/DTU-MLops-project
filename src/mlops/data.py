from __future__ import annotations

import typer
import kagglehub
import torch
from pathlib import Path
import csv
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

DATASET_HANDLE = "gpiosenka/cards-image-datasetclassification"

card_color = ["hearts", "diamonds", "clubs", "spades"]
card_color_to_idx = {color: idx for idx, color in enumerate(card_color)}

card_type = [ "joker", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "jack", "queen", "king"]
card_type_to_idx = {ctype: idx for idx, ctype in enumerate(card_type)}

def preprocess_data(processed_dir: str = "../../data/processed") -> None:
    # Download the dataset
    dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csv_path = dataset_path / "cards.csv"

    # Load the data
    splits = ["train", "valid", "test"]
    images = {split: [] for split in splits}
    labels = {split: [] for split in splits}

    valid_ext = {".jpg", ".jpeg", ".png"}

    with csv_path.open(newline="") as f:
        reader = list(csv.DictReader(f))
        for row in tqdm(reader, desc="Loading images"):
            img_path = dataset_path / row["filepaths"]

            if img_path.suffix.lower() not in valid_ext:
                print(f"Skipping unsupported file extension: {img_path}")
                continue

            split = row["data set"]
            image = read_image(str(img_path), mode=ImageReadMode.RGB)
            #image = image.float() / 255.0  # Normalize to [0, 1] # Converting here to float increases the size 4x

            class_name = row["labels"]
            if " of " in class_name:
                card_type, card_color = class_name.split(" of ", 1)
            else:
                card_type, card_color = class_name, "hearts"  # Default color for jokers

            card_type_idx = card_type_to_idx.get(card_type, -1)
            card_color_idx = card_color_to_idx.get(card_color, -1)

            images[split].append(image)
            labels[split].append(torch.tensor([card_type_idx, card_color_idx], dtype=torch.long))

    # Convert data into TensorDatasets
    datasets = {}
    for split in splits:
        images_tensor = torch.stack(images[split])
        labels_tensor = torch.stack(labels[split])
        datasets[split] = TensorDataset(images_tensor, labels_tensor)

    # Save the processed datasets
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        images, labels = datasets[split].tensors

        torch.save(
            {"images": images, "labels": labels},
            processed_dir / f"{split}.pt"
        )

def load_data(processed_dir: str = "../../data/processed", split: str = "train") -> TensorDataset:
    processed_dir = Path(processed_dir)
    data_path = processed_dir / f"{split}.pt"
    data = torch.load(data_path)
    images = data["images"]
    labels = data["labels"]
    return TensorDataset(images, labels)

if __name__ == "__main__":
    typer.run(preprocess_data)
