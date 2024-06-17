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


def _get_file_size_sum(files: pd.Series) -> int:
    """Get the size of a list of files in bytes."""
    return int(sum(Path(file).stat().st_size for file in files) / 1024 / 1024)


class HNNEClustering:
    """Useful methods of HNNE."""

    def __init__(
        self,
        csv_path: str = None,
        project_name: str = None,
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

        cols = self.df.columns
        wav_col = None
        for col in cols:
            if "wav" in col.lower():
                wav_col = col
                break
        if wav_col is None:
            raise ValueError("Please provide a column with wav file paths.")

        # Define attributes beforehand
        self.wav_col = wav_col
        self.project_name = project_name
        self.partition_levels = None
        self.cluster_folder_list = []
        self.original_df = self.df.copy()

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
        hierarchically: bool = True,
        clean_directory: bool = False,
    ) -> None:
        """Save audio files with their corresponding clusters."""

        # Calculate the size of data before saving it in the hard drive
        wav_col = self.wav_col
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

        save_folder = base_folder / self.project_name
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

                progress = f"[{i+1}/{len(cluster)}]"
                print(
                    f"\r>>> {style_it(color='green', style='bold', text=progress)} "
                    f"{src_wav.name} -> {save_cluster_folder}",
                    end="",
                )

            print("\n>>> Saving complete.")
            print()

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
                self.cluster_folder_list.append(save_cluster_folder)

                # Copy audio files from the original files
                src_wav = Path(self.df["wav"][i])
                shutil.copy(
                    src=self.df["wav"][i], dst=save_cluster_folder / src_wav.name
                )

                progress = f"[{i+1}/{len(cluster)}]"
                print(
                    f"\r>>> {style_it(color='green', style='bold', text=progress)} "
                    f"{src_wav.name} -> {save_cluster_folder}",
                    end="",
                )

            print("\n>>> Saving complete.")
            print()

        self.cluster_folder_list = list(set(self.cluster_folder_list))

    def confidence_cluster_by_cluster(
        self,
        model_name: str,
        hugging_face: object,
        csv_path: str,
        with_logits: str,
        orientation: Literal["max", "min"] = "max",
        check_percentile: int = 30,
    ) -> None:
        """Calculate confidence of each cluster by model."""

        # Compute confidences
        if (
            model_name == hugging_face.model_name
            and hugging_face.inference_result is not None
        ):
            print(
                f">>> Results from model {model_name} is already loaded.\n"
                ">>> Using the precomputed results."
            )
            print()

            confidences = hugging_face.inference_result.logits.detach().cpu()

        else:
            print(">>> Computing the confidence values...")
            print()

            hugging_face.load_model_for_confidence(model_name=model_name)

            confidences = hugging_face.compute_confidence(
                model_name=model_name,
                csv_path=csv_path,
            )

        confidences = with_logits(confidences=confidences)
        print(">>> Computing complete.")
        print()

        # Append it to the DataFrame
        print(">>> Modifying folder names with confidence values.")

        self.df["Confidences"] = confidences.tolist()

        self.cluster_folder_list = sorted(
            self.cluster_folder_list,
            key=lambda x: int(x.stem),
        )

        mean_confidence_list = []
        std_confidence_list = []
        new_cluster_folder_list = []

        for cluster_folder in self.cluster_folder_list:
            cluster_no = int(cluster_folder.stem)
            cluster_df = self.df.loc[self.df["Cluster_0"] == cluster_no]
            cluster_confidences = cluster_df["Confidences"][:]

            # Mean and std value across batch
            mean_confidence = cluster_confidences.mean(axis=0)
            std_confidence = cluster_confidences.std(axis=0)
            mean_confidence_list.append(mean_confidence)
            std_confidence_list.append(std_confidence)

            # Change folder name
            new_folder_name = (
                cluster_folder.parent
                / f"{cluster_folder.name}_m{mean_confidence:.2f}_s{std_confidence:.2f}"
            )
            cluster_folder.rename(new_folder_name)
            new_cluster_folder_list.append(new_folder_name)

        print(">>> Modification complete.")
        print()

        print(">>> Confidence values:")

        for i, cluster_folder in enumerate(new_cluster_folder_list):
            check_folder = (
                Path("./data/results/clustered_audio") / self.project_name / "check"
            )
            check_folder.mkdir(parents=True, exist_ok=True)

            text = (
                f"    - {cluster_folder.name}\t> "
                f"{str(cluster_folder.relative_to(Path('./data/results/clustered_audio')))}"
            )

            if orientation == "max":
                mean_confidence_np = np.array(mean_confidence_list)
                threshold = np.percentile(mean_confidence_np, check_percentile)

                if mean_confidence_list[i] < threshold:
                    color = "red"
                    style = "bold"

                    shutil.copytree(
                        src=cluster_folder, dst=check_folder / cluster_folder.name
                    )

                else:
                    color = style = None

            elif orientation == "min":
                mean_confidence_np = np.array(mean_confidence_list)
                threshold = np.percentile(mean_confidence_np, 100 - check_percentile)

                if mean_confidence_list[i] > threshold:
                    color = "red"
                    style = "bold"

                    shutil.copytree(
                        src=cluster_folder, dst=check_folder / cluster_folder.name
                    )

                else:
                    color = style = None

            else:
                raise ValueError("Please provide a valid orientation.")

            print(style_it(color=color, style=style, text=text))
        print()

    def remove_outliers(self) -> None:
        """Remove outliers from the check folder."""
        check_folder = (
            Path("./data/results/clustered_audio") / self.project_name / "check"
        )

        # Go remove normal files
        print(
            ">>> Check folder is created "
            f"> {style_it(color='red', style='bold', text=check_folder)}\n"
            ">>> Please check the folder for the clusters and remove the ones "
            "you do not want to reject."
        )
        print()

        print(
            f">>> Press {style_it(color='green', style='bold', text='Enter')} "
            "to continue when you are done."
        )
        input()

        # Make as csv file with rejection aimed files
        print(">>> Rejecting files will be listed in csv file.")

        wav_files = list(check_folder.rglob("*.wav"))
        df = self.original_df.copy()
        df["wav_name"] = df[self.wav_col].apply(lambda x: Path(x).name)

        new_df_list = []
        for wav_file in wav_files:

            new_df_list.append(df.loc[df["wav_name"] == wav_file.name])

        new_df = pd.concat(objs=new_df_list, axis=0)
        new_df.drop(columns=["wav_name"], inplace=True)

        saved_folder = str(
            Path("./data/results/clustered_audio/")
            / self.project_name
            / f"{self.csv_name}_rejected.csv"
        )

        new_df.to_csv(
            path_or_buf=saved_folder,
            index=False,
        )

        print(
            ">>> Rejected files are saved in "
            f"{style_it(color='red', style='bold', text=saved_folder)}."
        )
        print()
