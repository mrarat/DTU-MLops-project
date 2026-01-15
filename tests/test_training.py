import torch
import mlops.train as train_mod
from mlops.train import train as train_fn
from omegaconf import DictConfig, OmegaConf


def test_train_saves_model(monkeypatch, tmp_path):
    # Patch wandb to avoid external calls
    monkeypatch.setattr(train_mod.wandb, "init", lambda *a, **k: None)
    monkeypatch.setattr(train_mod.wandb, "log", lambda *a, **k: None)
    monkeypatch.setattr(train_mod.wandb, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(train_mod.wandb, "Artifact", lambda *a, **k: type('MockArtifact', (), {'add_file': lambda self, f: None})())

    # Use a tiny dataset (1 sample)z to keep training fast
    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            img = torch.zeros((3, 224, 224))  # dummy image
            target = torch.tensor([0, 0])  # dummy rank/suit
            return img, target

    monkeypatch.setattr(train_mod, "load_data", lambda processed_dir=None, split=None: TinyDataset())

    # Patch get_original_cwd to tmp_path
    monkeypatch.setattr(train_mod, "get_original_cwd", lambda: str(tmp_path))

    # Mock torch.save to avoid writing files
    calls = {}
    monkeypatch.setattr(train_mod.torch, "save", lambda obj, path: calls.setdefault("path", path))

    cfg = OmegaConf.create(
        {
        "wandb": 
            {"project": "test", "entity": "mlops-group-42"}, 
        "hyperparameters": 
            {"batch_size": 32, "epochs": 1, "lr": 0.001, "seed": 42}
        }
    )
    
    train_fn(cfg)

    # Assert that torch.save was called with the expected path
    expected_path = str(tmp_path / "models" / "model.pth")
    assert calls["path"] == expected_path
    assert not (tmp_path / "models" / "model.pth").exists()
