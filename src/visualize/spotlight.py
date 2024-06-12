"""Visualize data through Renumics Spotlight"""

# External
import pandas as pd

# Spotlight
from renumics.spotlight import Audio, Embedding, show


class SpotlightVisual:
    """Prepare data for Renumics Spotlight and visualize it."""

    def __init__(
        self,
        df: pd.DataFrame,
    ):
        # TODO: Add our model inference process & add prediction and true columns to df

        self.df = df
        self.dtypes = {
            "wav": Audio,
            "Embeddings": Embedding,
        }

    def visualize(self):
        """Visualize data through Renumics Spotlight."""

        show(
            dataset=self.df,
            dtype=self.dtypes,
        )
