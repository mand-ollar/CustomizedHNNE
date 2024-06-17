"""Make tsv files corresponding to the wav files."""

# Internal
from pathlib import Path
from typing import Literal

# External
import soundfile as sf

# Project
from utils.preprocess import resample


def tsv_from_wav(
    wav_path: str,
    label: str,
    hard_label_mode: Literal["full"],
) -> None:
    """Applied for one file."""
    # Path
    wav_path = Path(wav_path)
    tsv_path = wav_path.with_suffix(".tsv")

    # Read wav file
    audio, sr = sf.read(file=wav_path)

    if sr != 16000:
        audio, sr = resample(audio=audio, orig_sr=sr, targ_sr=16000)

    en_pnt = audio.shape[0] - 1

    # Mode
    if hard_label_mode == "full":
        st_pnt = 0

    # Tsv
    with open(tsv_path, "w", encoding="ascii") as f:
        f.write(f"{st_pnt}\t{en_pnt}\t{label}\n")
        f.close()


def tsv_from_path(
    data_path: str,
    label: str,
    hard_label_mode: Literal["full"],
) -> None:
    """Applied for all files in the directory."""
    # Path
    data_path = Path(data_path)

    # Get wav files
    wav_files = list(data_path.glob("*.wav"))

    for i, wav_file in enumerate(wav_files):
        tsv_from_wav(
            wav_path=wav_file,
            label=label,
            hard_label_mode=hard_label_mode,
        )
