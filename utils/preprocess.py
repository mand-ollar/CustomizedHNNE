"""Preprocess audio data"""

import random

import torch
import torch.nn.functional as F
import torchaudio


def pad_or_truncate(audio: torch.tensor, max_sec: int = 2) -> torch.Tensor:
    """To make audio data to have the same length.
    Output audio shape: (1, len)"""

    assert audio.type() != torch.FloatTensor, "Audio must be a torch.FloatTensor"

    audio = _format_audio(audio=audio)

    # Pad or truncate
    if max_sec is None:
        edited_audio = audio

    elif audio.size(1) < max_sec * 16000:
        diff = max_sec * 16000 - audio.size(1)
        front = random.randint(0, diff)
        back = diff - front

        edited_audio = F.pad(
            input=audio, pad=(front, back, 0, 0), mode="constant", value=0
        )

    elif audio.size(1) > max_sec * 16000:
        diff = audio.size(1) - max_sec * 16000
        front = random.randint(0, diff)
        back = diff - front

        edited_audio = audio[:, front : audio.size(0) - back]

    else:
        edited_audio = audio

    return edited_audio


def resample(audio: torch.tensor, orig_sr: int, targ_sr: int):
    """resample audio data"""
    audio = _format_audio(audio=audio)

    if orig_sr == targ_sr:
        return audio, targ_sr

    else:
        edited_audio = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=targ_sr
        )(audio)

        return edited_audio, targ_sr


def _format_audio(audio: torch.tensor):
    """Format audio data"""

    if len(audio.size()) == 1:
        audio = audio.unsqueeze(dim=0)

    elif len(audio.size()) == 2:
        # Set size format
        if audio.size(0) > audio.size(1):
            audio = audio.t()

        # Stereo to mono
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)

    return audio


def load_audio(audio_path: str):
    """Load audio file
    Shape: [channel, time]"""
    audio, sr = torchaudio.load(uri=audio_path)

    return audio, sr
