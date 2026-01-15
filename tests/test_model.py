from mlops.model import Model
import torch


def test_model():
    model = Model()
    x = torch.randn((1, 3, 224, 224))
    y = model(x)
    assert y["suit"].shape == (1, 4), "Suit output not correct size"
    assert y["rank"].shape == (1, 13), "Rank utput not correct size"
