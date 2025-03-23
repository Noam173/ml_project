import pandas as pd


def plot_training(history):
    data = pd.DataFrame(history).shift(1)

    data.plot(y=["accuracy", "val_accuracy"])

    data.plot(y=["loss", "val_loss"])
