""" From wav data directory, make csv data file. """

# Internal
from pathlib import Path


def make_csv(
    data_path: str,
    save_path: str,
    how_to_label: str = "directory",
):
    """Make csv file from wav data directory.

    Args:
        data_path (str): Path to wav data directory.
        save_path (str): Path to save csv file."""

    data_path = Path(data_path)
    save_path = Path(save_path)

    wavs = data_path.rglob("*.wav")

    csv_data = ["wav,label"]
    for wav in wavs:
        if how_to_label == "directory":
            label = wav.parent.name
        else:
            label = 0

        csv_data.append(f"{wav.absolute()},{label}")

    with open(file=save_path, mode="w", encoding="ascii") as f:
        f.write("\n".join(csv_data))


if __name__ == "__main__":
    make_csv(
        data_path="/data/minseok/data_share/AggressiveYell/AggressiveYell_16K",
        save_path="./data/csv/AggressiveYell.csv",
    )
