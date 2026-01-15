from torch import nn
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class Model(nn.Module):
    def __init__(
        self,
        num_ranks: int = 13,
        num_suits: int = 4,
        weights: MobileNet_V3_Small_Weights = MobileNet_V3_Small_Weights.DEFAULT,
        freeze_features: bool = True,
    ):
        super().__init__()

        self.weights = weights
        self.preset = weights.transforms()
        self.mean = self.preset.mean
        self.std = self.preset.std

        self.model = mobilenet_v3_small(weights=weights)

        if freeze_features:
            self.freeze_features()

        # MobileNetV3 classifier ends with Linear(1024 -> 1000) for ImageNet.
        # Replace ONLY that last layer with Identity so the model outputs 1024-d embeddings.
        in_f = self.model.classifier[-1].in_features  # typically 1024
        self.model.classifier[-1] = nn.Identity()

        # Two heads from the 1024 embedding
        self.rank_head = nn.Linear(in_f, num_ranks)
        self.suit_head = nn.Linear(in_f, num_suits)

    def freeze_features(self) -> None:
        for p in self.model.features.parameters():
            p.requires_grad = False

    def unfreeze_features(self) -> None:
        for p in self.model.features.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor):
        emb = self.model(x)  # shape [B, 1024] after replacing last layer with Identity
        return {
            "rank": self.rank_head(emb),  # logits [B, 13]
            "suit": self.suit_head(emb),  # logits [B, 4]
        }


if __name__ == "__main__":
    model = Model()
    # print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.rand(1, 3, 224, 224)
    out = model(dummy_input)
    print("Rank logits shape:", out["rank"].shape)
    print("Suit logits shape:", out["suit"].shape)
