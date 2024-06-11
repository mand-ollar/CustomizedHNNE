"""Make feature extractor as a nn.module."""

# Torch
import torch
import torch.nn as nn

# Hugging Face
from transformers import AutoFeatureExtractor


class FeatureExtractorModule(nn.Module):
    """Feature extractor module."""

    def __init__(
        self,
        model_name: str,
        sampling_rate: int = 16000,
    ):
        super().__init__()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir="./model",
            force_download=True,
            return_tensors="pt",
        )

        self.sampling_rate = sampling_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(
        self,
        x: torch.tensor,
    ):
        """Forward pass."""

        x = self.feature_extractor(
            x,
            sampling_rate=self.sampling_rate,
        )
        x = torch.tensor(x["input_values"][0])
        x = x.to(self.device)

        return x
