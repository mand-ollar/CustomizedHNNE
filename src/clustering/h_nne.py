""" Hierarchical Nearest Neighbour Graph Embedding (HNNE) algorithm. """

# Internal
from typing import Literal

# External
import numpy as np
from hnne import HNNE


class HNNEClustering:
    """Useful methods of HNNE."""

    def __init__(
        self,
        dim: int = 2,
        metric: Literal[
            "cosine", "euclidean", "cityblock", "l1", "l2", "manhattan"
        ] = "cosine",
        radius: float = 0.45,
        ann_threshold: int = 40000,
        preliminary_embedding: Literal["pca", "pca_centroids", "random_linear"] = "pca",
    ):
        self.hnne = HNNE(
            dim=dim,
            metric=metric,
            radius=radius,
            ann_threshold=ann_threshold,
            preliminary_embedding=preliminary_embedding,
        )

    def cluster_it(
        self,
        embeddings: np.array,
    ):
        """Cluster embeddings with HNNE."""

        projection = self.hnne.fit_transform(X=embeddings)
        print(projection)
