import shutil
from gc import collect

import pandas as pd
import tensorflow as tf
from Reset_data import create_directory
from test_model import create_model as model


def split_data(train_path_csv: str, path) -> str:
    """
    Parameters
    ----------
    train_path : string.
        the path to the original dataset's DataFrame file to prepare it for splitting.

    Returns
    -------
    None.

    """
    dataset_path, flag = create_directory(path)

    if flag:
        path = pd.read_csv(train_path_csv)

        real = path[path["label"] == 0].file_name
        ai = path[path["label"] == 1].file_name

        for i in real:
            shutil.copy(f"{dataset_path}/{i}", f"{dataset_path}/classes/real")

        for i in ai:
            shutil.copy(f"{dataset_path}/{i}", f"{dataset_path}/classes/ai")

    collect()

    return f"{dataset_path}/classes"


def preprocess_data(train_path: str, batch_size: int) -> None:
    data = tf.keras.utils.image_dataset_from_directory(
        train_path, shuffle=True, seed=0, image_size=(224, 224), batch_size=batch_size
    ).map(lambda x, y: (x / 255.0, y))

    size = len(data)

    val_size = int(size * 0.3)

    train = data.skip(val_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val = data.take(val_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train, val
