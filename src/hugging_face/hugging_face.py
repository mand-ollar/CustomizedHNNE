"""This module loads a pre-trained model from Hugging Face's model hub."""

# Internal
import platform
from pathlib import Path
from typing import Literal

# External
from tqdm import tqdm

# Hugging Face
from transformers import AutoModel, AutoModelForAudioClassification

# Torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project
from src.custom_dataset import CustomDataset
from utils.style import style_it


class HuggingFace:
    """Load pre-trained model and compute embeddings."""

    def __init__(
        self,
        model_name: str,
        sampling_rate: int = 16000,
        max_sec: int = None,
        batch_size: int = 1,
        num_proc: int = 0,
    ):
        # Related to model
        self.model = None
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_proc = num_proc

        # Set number of processors for Mac, Linux
        if platform.system() == "Darwin":
            print(">>> Running on MacOS. Setting num_proc=0.")
            num_proc = 0
        elif platform.system() == "Linux":
            print(f">>> Running on Linux. Setting num_proc={num_proc}.")
        else:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

        # Related to audio
        self.sampling_rate = sampling_rate
        self.max_sec = max_sec

        # Define self attrtibute beforehand
        self.inference_result = None

    def _load_model(
        self,
        model_name: str,
        mode: Literal["confidence", "embedding"] = "embedding",
    ) -> nn.Module:
        """Combine feature extractor and model from Hugging Face as a nn.Module."""

        Path("./model").mkdir(parents=True, exist_ok=True)

        print(">>> Loading model from Hugging Face...")

        if mode == "embedding":
            model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_name,
                cache_dir="./model",
                force_download=True,
            )
        elif mode == "confidence":
            model = AutoModelForAudioClassification.from_pretrained(
                pretrained_model_name_or_path=model_name,
                cache_dir="./model",
                force_download=True,
            )
        else:
            raise NotImplementedError(f"Mode {mode} is not supported.")

        model = model.to(self.device).eval()

        if model_name == self.model_name:
            self.model = model

        print(
            f">>> Model {style_it(color='red', style='bold', text=model_name)} "
            "loading is completed.\n"
            f"    - Sampling rate: {self.sampling_rate} Hz\n"
            f"    - Max length: {self.max_sec} sec\n",
        )

        return model

    def compute_embeddings(
        self,
        csv_path: str,
        output_type: Literal["numpy", "torch"] = "torch",
    ) -> torch.tensor:
        """Compute embeddings with self.model."""

        # Load Hugging Face model
        self._load_model(self.model_name)

        # Dataset & DataLoader
        print(">>> Setting up DataLoader...")
        dataset = CustomDataset(
            data_path=csv_path,
            model_name=self.model_name,
            sr=self.sampling_rate,
            max_sec=self.max_sec,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_proc,
            shuffle=False,
        )
        print(">>> DataLoader is ready.")
        print()

        # Collect embeddings in the list
        print(f">>> Computing embeddings with {self.model_name} on {self.device.type}.")

        embeddings = []

        with torch.no_grad():
            for x, _ in tqdm(
                dataloader,
                ncols=100,
                desc="Computing embeddings",
                leave=False,
            ):
                x = x.to(self.device)
                self.inference_result = self.model(x)  # For confidence computation
                embedding = self.inference_result.last_hidden_state[:, 0].detach().cpu()
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

        print(
            ">>> Embeddings extraction is completed.\n"
            f"    - Output type: {output_type}"
        )
        print()

        return embeddings

    def load_model_for_confidence(
        self,
        model_name: str,
    ):
        """Load model for confidence computation."""

        self.model = self._load_model(
            model_name=model_name,
            mode="confidence",
        )

    def compute_confidence(
        self,
        model_name: str,
        csv_path: str,
    ):
        """Compute confidence from the inference result."""

        # # Make it as csv
        # csv_save_path = Path("./.cache")
        # csv_save_path.mkdir(parents=True, exist_ok=True)
        # make_csv(data_path=data_path, save_path=csv_save_path / f"{project_name}.csv")

        # Dataset & DataLoader
        dataset = CustomDataset(
            data_path=csv_path,
            model_name=model_name,
            sr=self.sampling_rate,
            max_sec=self.max_sec,
            print_label_mappings=False,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_proc,
            shuffle=False,
        )

        confidences = []

        with torch.no_grad():
            for x, _ in tqdm(
                dataloader,
                ncols=100,
                desc="Computing confidences",
                leave=False,
            ):
                x = x.to(self.device)
                self.inference_result = self.model(x)
                confidence = self.inference_result.logits.detach().cpu()
                confidences.append(confidence)

        # Concatenate confidences from the list of tensors
        confidences = torch.cat(confidences, dim=0)

        return confidences
