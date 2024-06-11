"""Main code of this repository."""

from src.hugging_face.hugging_face import HuggingFace
from src.clustering import h_nne

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

hugging_face = HuggingFace(
    model_name=MODEL_NAME,
    sampling_rate=16000,
    max_sec=None,
)

emb = hugging_face.compute_embeddings(
    data_path="./data/csv/CREMA-D.csv",
    batch_size=4,
    output_type="numpy",
)

HN = h_nne.HNNEClustering()
HN.cluster_it(embeddings=emb)
