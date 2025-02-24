import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training(history):
    data = pd.DataFrame.from_dict(history)
    data["epochs"] = data.index + 1

    sns.lineplot(x=data.epochs, y=data.accuracy, label="Train accuracy")
    sns.lineplot(x=data.epochs, y=data.val_accuracy, label="Val accuracy")
    plt.show()

    sns.lineplot(x=data.epochs, y=data.loss, label="Train loss")
    sns.lineplot(x=data.epochs, y=data.val_loss, label="Val loss")
    plt.show()
