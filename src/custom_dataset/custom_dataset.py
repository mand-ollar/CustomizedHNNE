""" This module contains the CustomDataset class
which is used to create a custom dataset for the model.
Extract features with the information of Hugging Face model name. """

# Internal
from pathlib import Path

# External
import pandas as pd

# Torch
from torch.utils.data import Dataset

# Hugging Face
from transformers import AutoFeatureExtractor

# Project
from utils.preprocess import load_audio, resample, pad_or_truncate


class CustomDataset(Dataset):
    """Dataset class for custom dataset."""

    def __init__(
        self,
        data_path: str,
        model_name: str,
        max_sec: int = None,
        sr: int = 16000,
    ):
        super().__init__()

        data_path = Path(data_path)

        Path("./model").mkdir(parents=True, exist_ok=True)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir="./model",
            force_download=True,
            return_tensors="pt",
        )

        """ CSV file format:
        | wav | label | """
        csv_file = data_path
        df = pd.read_csv(filepath_or_buffer=csv_file)

        cols = df.columns
        for col in cols:
            if "wav" in col.lower():
                wav_col = col
            elif "label" in col.lower():
                label_col = col

        self.wavs = df[wav_col]
        self.labels = df[label_col]
        self.unique_labels = self.labels.unique()
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.max_sec = max_sec
        self.sr = sr

        print("\n>>> Label Mappings:")
        print(
            *[f"    - {i:02} {label}" for label, i in self.label_to_idx.items()],
            sep="\n",
        )

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx) -> tuple:

        # Load label
        label = self.labels[idx]
        idx_label = self.label_to_idx[label]

        # Load audio
        wav_path = self.wavs[idx]
        audio, sr = load_audio(audio_path=wav_path)

        # Preprocess
        audio, sr = resample(audio=audio, orig_sr=sr, targ_sr=self.sr)
        audio = pad_or_truncate(audio=audio, max_sec=self.max_sec)

        # Extract features
        audio = audio.squeeze(dim=0)
        feature = self.feature_extractor(
            audio,
            sampling_rate=self.sr,
            return_tensors="pt",
        )
        feature = feature["input_values"][0].clone().detach()

        return feature, idx_label
