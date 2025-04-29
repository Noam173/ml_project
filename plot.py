import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def plot_training(history: dict) -> None:
    data = pd.DataFrame(history).shift(1)
    data.columns = data.columns.str.lower()

    data.plot(y=["accuracy", "val_accuracy"])

    data.plot(y=["loss", "val_loss"])


def plot_con_matrix(matrix: np.ndarray) -> None:
    matrix_df = pd.DataFrame(matrix, columns=["ai", "real"])

    matrix_df["predicted labels"] = "ai", "real"
    matrix_df.set_index("predicted labels", inplace=True)
    sns.heatmap(matrix_df, annot=True, fmt="d").set_xlabel("real labels")
    plt.show()


def plot_images(pred: np.ndarray, files: tf.data.Dataset) -> None:
    for i, file in enumerate(files):
        file = file.numpy().decode("utf-8")
        plt.imshow(plt.imread(file))
        label = "real" if pred.round()[i] == 1 else "ai"
        plt.title(f'predicted {label}')
        plt.axis('off')
        plt.show()
        print("=" * 100)
