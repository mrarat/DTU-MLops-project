from torch.utils.data import Dataset
from pathlib import Path

from mlops.data import *


def test_data(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed_test"
    preprocess_data(processed_dir=processed_dir)
    processed_path = Path(processed_dir)
    assert processed_path.exists()
    assert (processed_path / "train.pt").exists()
    assert (processed_path / "valid.pt").exists()
    assert (processed_path / "test.pt").exists()

    train_ds = load_data(processed_dir=processed_dir, split="train")
    valid_ds = load_data(processed_dir=processed_dir, split="valid")
    test_ds = load_data(processed_dir=processed_dir, split="test")

    assert isinstance(train_ds, Dataset)
    assert isinstance(valid_ds, Dataset)
    assert isinstance(test_ds, Dataset)

    assert train_ds.tensors[0].shape[1:] == (3, 224, 224)
    assert valid_ds.tensors[0].shape[1:] == (3, 224, 224)
    assert test_ds.tensors[0].shape[1:] == (3, 224, 224)

    assert train_ds.tensors[1].shape[1] == 2  # Two labels: rank and suit
    assert valid_ds.tensors[1].shape[1] == 2  # Two labels: rank and suit
    assert test_ds.tensors[1].shape[1] == 2  # Two labels: rank and suit
