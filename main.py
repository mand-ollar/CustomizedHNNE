"""Main code of this repository."""

# Importing time
import time

print(">>> Importing modules...")
st = time.time()
# pylint: disable=wrong-import-position

# Internal
import warnings
from pathlib import Path

# External
import numpy as np
from tqdm import tqdm

# Torch
import torch
import torch.nn.functional as F

# Project
# pylint: disable=unused-import

from src.hugging_face.hugging_face import HuggingFace
from src.clustering import h_nne
from src.visualize import spotlight

# pylint: enable=unused-import
# pylint: enable=wrong-import-position

############################## PREPARATION ##############################
# Not showing warnings
warnings.filterwarnings("ignore")


# Not leaving from tqdm progress bar
def _new_init(self, *args, **kwargs):
    kwargs.setdefault("leave", False)
    kwargs.setdefault("ncols", 100)
    _original_init(self, *args, **kwargs)


_original_init = tqdm.__init__
tqdm.__init__ = _new_init
##########################################################################


################################# CONFIG #################################
print(">>> Modules imported.")
print(f">>> {time.time() - st:.2f} seconds elapsed for module imports.")
print()

# CONFIG
MODEL_LIST = [
    "MIT/ast-finetuned-audioset-10-10-0.4593",  # 0
    "facebook/wav2vec2-base-960h",  # 1
    "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",  # 2
    "audeering/wav2vec2-large-robust-24-ft-age-gender",  # 3
    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",  # 4
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",  # 5
]
MODEL_NAME = MODEL_LIST[2]
CSV_FILE = "./data/csv/AngryMedia_female.csv"
PROJECT_NAME = "AngryMedia_gender_classification"
COMPUTE_EMBEDDINGS = False
##########################################################################


################################## TOOLS #################################
hugging_face = HuggingFace(
    model_name=MODEL_NAME,
    sampling_rate=16000,
    max_sec=5,
    batch_size=64,
    num_proc=12,
)
if COMPUTE_EMBEDDINGS:
    emb = hugging_face.compute_embeddings(
        csv_path=CSV_FILE,
        output_type="numpy",
    )

    np.save(file=f"./data/results/{Path(CSV_FILE).stem}_embed.npy", arr=emb)

else:
    emb = np.load(file=f"./data/results/{Path(CSV_FILE).stem}_embed.npy")

HN = h_nne.HNNEClustering(
    csv_path=CSV_FILE,
    project_name=PROJECT_NAME,
)
HN.cluster_it(embeddings=emb)
HN.save_it(
    hierarchically=True,
    clean_directory=True,
)


def _logits_callback(confidences: torch.tensor) -> torch.tensor:
    """Post processing with raw logits."""
    output = F.sigmoid(confidences)[:, 0]

    return output


HN.confidence_cluster_by_cluster(
    model_name=MODEL_LIST[2],
    hugging_face=hugging_face,
    csv_path=CSV_FILE,
    with_logits=_logits_callback,
)
HN.remove_outliers()
