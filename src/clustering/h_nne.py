""" Hierarchical Nearest Neighbour Graph Embedding (HNNE) algorithm. """

# Internal
from pathlib import Path
from typing import Literal

# External
import numpy as np
import pandas as pd
from hnne import HNNE


class HNNEClustering:
    """Useful methods of HNNE."""

    def __init__(
        self,
        csv_path: str = None,
        dim: int = 2,
        metric: Literal[
            "cosine", "euclidean", "cityblock", "l1", "l2", "manhattan"
        ] = "cosine",
        radius: float = 0.45,
        ann_threshold: int = 40000,
        preliminary_embedding: Literal["pca", "pca_centroids", "random_linear"] = "pca",
    ):
        # Chceck csv_file
        assert csv_path is not None, "Please provide a csv file path."
        self.csv_name = Path(csv_path).stem

        # Initialize HNNE
        self.hnne = HNNE(
            dim=dim,
            metric=metric,
            radius=radius,
            ann_threshold=ann_threshold,
            preliminary_embedding=preliminary_embedding,
        )

        # Make pd.DataFrame corresponding to embeddings
        with open(file=csv_path, mode="r", encoding="ascii") as file:
            self.df = pd.read_csv(filepath_or_buffer=file)

    def cluster_it(
        self,
        embeddings: np.array,
    ) -> pd.DataFrame:
        """Cluster embeddings with HNNE."""

        # {self.dim} dimensional embeddings
        projection = self.hnne.fit_transform(X=embeddings)

        # partitions: np.array
        # Shape: (n_samples, n_partitions)
        partitions = self.hnne.hierarchy_parameters.partitions

        # partition_sizes: list
        partition_sizes = self.hnne.hierarchy_parameters.partition_sizes
        partition_levels = len(partition_sizes)

        # Append partitions to self.df
        partitions_df = pd.DataFrame(
            data={
                f"Cluster_{i}": partitions[:, i].astype("int32")
                for i in range(partition_levels)
            },
            dtype="int32",
        )
        self.df = pd.concat(objs=[self.df, partitions_df], axis=1)
        # Convert to int32
        self.df = self.df.astype(
            {f"Cluster_{i}": "int32" for i in range(partition_levels)}
        )

        # Append projected vectors to self.df
        projection_df = pd.DataFrame(
            data={"Projection": projection.tolist()},
        )
        self.df = pd.concat(objs=[self.df, projection_df], axis=1)

        # Append embeddings to self.df
        embeddings_df = pd.DataFrame(
            data={"Embeddings": embeddings.tolist()},
        )
        self.df = pd.concat(objs=[self.df, embeddings_df], axis=1)

        # Save pd.DataFrame
        Path("./data/results/clustered").mkdir(parents=True, exist_ok=True)

        self.df.to_csv(
            path_or_buf=f"./data/results/clustered/{self.csv_name}.csv",
            index=False,
        )
        self.df.to_json(
            path_or_buf=f"./data/results/clustered/{self.csv_name}.json",
            orient="records",
            indent=2,
        )

        return self.df
