""" Hierarchical Nearest Neighbour Graph Embedding (HNNE) algorithm. """

# Internal
import shutil
from pathlib import Path
from typing import Literal

# External
import numpy as np
import pandas as pd
from hnne import HNNE

# Project
from utils.style import style_it


def _get_folder_size(folder: Path) -> int:
    """Get the size of a folder in bytes."""
    assert folder.is_dir(), "Please provide a valid folder."

    return sum(f.stat().st_size for f in folder.glob("**/*") if f.is_file())


def _get_file_size_sum(files: pd.Series) -> int:
    """Get the size of a list of files in bytes."""
    return int(sum(Path(file).stat().st_size for file in files) / 1024 / 1024)


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
        print(">>> HNNE initialized with")
        print(f"    - dim: {dim}")
        print(f"    - metric: {metric}")
        print(f"    - radius: {radius}")
        print(f"    - ann_threshold: {ann_threshold}")
        print(f"    - preliminary_embedding: {preliminary_embedding}")
        print()

        # Make pd.DataFrame corresponding to embeddings
        with open(file=csv_path, mode="r", encoding="ascii") as file:
            self.df = pd.read_csv(filepath_or_buffer=file)

        # Define attributes beforehand
        self.partition_levels = None
        self.cluster_folder_list = []

    def cluster_it(
        self,
        embeddings: np.array,
    ) -> pd.DataFrame:
        """Cluster embeddings with HNNE."""

        # {self.dim} dimensional embeddings
        print(">>> Start clustering with HNNE...")
        projection = self.hnne.fit_transform(X=embeddings)
        print(">>> Clustering complete.")

        # partitions: np.array
        # Shape: (n_samples, n_partitions)
        partitions = self.hnne.hierarchy_parameters.partitions

        # partition_sizes: list
        partition_sizes = self.hnne.hierarchy_parameters.partition_sizes
        partition_levels = len(partition_sizes)

        print(f"    - Number of partitions: {partition_levels}")
        print(f"    - Partition sizes: {partition_sizes}")
        print()

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

        self.partition_levels = partition_levels

        return self.df

    def save_it(
        self,
        project_name: str,
        hierarchically: bool = True,
        clean_directory: bool = False,
    ) -> None:
        """Save audio files with their corresponding clusters."""

        # Calculate the size of data before saving it in the hard drive
        cols = self.df.columns
        wav_col = None
        for col in cols:
            if "wav" in col.lower():
                wav_col = col
                break
        if wav_col is None:
            raise ValueError("Please provide a column with wav file paths.")

        size_in_meg_bytes = _get_file_size_sum(files=self.df[wav_col])

        # Setting folder paths
        if size_in_meg_bytes > 1024 * 5:
            base_folder = Path("/data/minseok/results/clustered_audio")
            print(
                f">>> Total data size is {size_in_meg_bytes} MB. Saving in SSD drive."
            )
        else:
            base_folder = Path("./data/results/clustered_audio")
            print(
                f">>> Total data size is {size_in_meg_bytes} MB. Saving in Hard drive."
            )

        save_folder = base_folder / project_name
        save_folder.mkdir(parents=True, exist_ok=True)
        print(f">>> Saving in {save_folder}.")
        print()

        # If you want to clean the directory
        if clean_directory:
            print(
                ">>> Cleaning directory is "
                f"{style_it(color='green', style='bold', text='enabled')}."
            )
            print(
                ">>> Directory "
                f"{style_it(color='default', style='bold', text=base_folder.absolute())} "
                "is being cleaned."
            )
            shutil.rmtree(path=base_folder, ignore_errors=True)
            print(">>> Cleaning complete.")
            print()

        # Save audio files
        if hierarchically:
            print(
                ">>> Hierarchical saving is "
                f"{style_it(color='green', style='bold', text='enabled')}."
            )

            # On the level that has smallest clusters
            cluster = self.df["Cluster_0"]
            self.cluster_folder_list = []

            for i, j in enumerate(cluster):
                mid_folders = []
                for k in range(1, self.partition_levels)[::-1]:
                    mid_folders.append(str(self.df[f"Cluster_{k}"][i]))

                save_cluster_folder = save_folder / f"{'/'.join(mid_folders)}/{j}"
                save_cluster_folder.mkdir(parents=True, exist_ok=True)
                self.cluster_folder_list.append(save_cluster_folder)

                # Copy audio files from the original files
                src_wav = Path(self.df["wav"][i])
                shutil.copy(
                    src=self.df["wav"][i], dst=save_cluster_folder / src_wav.name
                )
                print(
                    f"\r>>> {i+1}/{len(cluster)}: {src_wav.name} -> {save_cluster_folder}",
                    end="",
                )
            print("\n")

        else:
            print(
                ">>> Hierarchical saving is "
                f"{style_it(color='red', style='bold', text='disabled')}."
            )

            # On the level that has largest clusters
            cluster = self.df[f"Cluster_{self.partition_levels - 1}"]

            for i, j in enumerate(cluster):
                save_cluster_folder = save_folder / str(j)
                save_cluster_folder.mkdir(parents=True, exist_ok=True)

                # Copy audio files from the original files
                src_wav = Path(self.df["wav"][i])
                shutil.copy(
                    src=self.df["wav"][i], dst=save_cluster_folder / src_wav.name
                )
                print(
                    f"\r>>> {i+1}/{len(cluster)}: {src_wav.name} -> {save_cluster_folder}",
                    end="",
                )
            print("\n")

    def confidence_cluster_by_cluster(self, model_name: str) -> None:
        """Calculate confidence of each cluster by model."""
