import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_training(history: dict) -> None:
    data = pd.DataFrame(history).shift(1)

    data.plot(y=["accuracy", "val_accuracy"])

    data.plot(y=["loss", "val_loss"])


def plot_con_matrix(matrix: np.ndarray) -> None:
    matrix_df = pd.DataFrame(matrix, columns=["positive", "negative"])

    matrix_df["predicted labels"] = "positive", "negative"
    matrix_df.set_index("predicted labels", inplace=True)
    sns.heatmap(matrix_df.T, annot=True, fmt="d").set_ylabel("real labels")
    plt.show()
