import pandas as pd


def plot_training(history):
    data = pd.DataFrame.from_dict(history)
    data["epochs"] = data.index + 1

    data.plot(x="epochs", y=["accuracy", "val_accuracy"])

    data.plot(x="epochs", y=["loss", "val_loss"])
