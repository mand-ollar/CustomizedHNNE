"""This module loads a pre-trained model from Hugging Face's model hub."""

# Internal
import platform
from pathlib import Path
from typing import Literal

# External
from tqdm import tqdm

# Hugging Face
from transformers import AutoModel

# Torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project
from src.custom_dataset import CustomDataset


class HuggingFace:
    """Load pre-trained model and compute embeddings."""

    def __init__(
        self,
        model_name: str,
        sampling_rate: int = 16000,
        max_sec: int = None,
    ):
        # Related to model
        self.model = None
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Related to audio
        self.sampling_rate = sampling_rate
        self.max_sec = max_sec

    def _load_model(self) -> nn.Module:
        """Combine feature extractor and model from Hugging Face as a nn.Module."""

        Path("./model").mkdir(parents=True, exist_ok=True)

        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            cache_dir="./model",
            force_download=True,
        )
        self.model = self.model.to(self.device).eval()

        msg = (
            f"\nModel {self.model_name} loading is completed.\n",
            f"- Sampling rate: {self.sampling_rate} Hz\n",
            f"- Max length: {self.max_sec} sec\n",
        )
        print(*msg)

        return self.model

    def compute_embeddings(
        self,
        data_path: str,
        batch_size: int = 1,
        num_proc: int = 0,
        output_type: Literal["numpy", "torch"] = "torch",
    ) -> torch.tensor:
        """Compute embeddings with self.model."""

        # Set number of processors for Mac, Linux
        if platform.system() == "Darwin":
            num_proc = 0
        elif platform.system() == "Linux":
            pass
        else:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

        # Load Hugging Face model
        self._load_model()

        # Dataset & DataLoader
        dataset = CustomDataset(
            data_path=data_path,
            model_name=self.model_name,
            sr=self.sampling_rate,
            max_sec=self.max_sec,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_proc,
            shuffle=False,
        )

        # Collect embeddings in the list
        print(f"Computing embeddings with {self.model_name} on {self.device.type}.")

        embeddings = []

        with torch.no_grad():
            for x, _ in tqdm(
                dataloader,
                ncols=100,
                desc="Computing embeddings",
                leave=False,
            ):
                x = x.to(self.device)
                embedding = self.model(x).last_hidden_state[:, 0].detach().cpu()
                embeddings.append(embedding)

        # Concatenate embeddings from the list of tensors
        embeddings = torch.cat(embeddings, dim=0)

        # Output type numpy.array or torch.tensor
        if output_type == "numpy":
            embeddings = embeddings.numpy()
        elif output_type == "torch":
            pass
        else:
            raise NotImplementedError(f"Output type {output_type} is not supported.")

        return embeddings
