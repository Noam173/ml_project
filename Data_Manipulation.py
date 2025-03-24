import shutil
from gc import collect
from glob import glob
import pandas as pd
import tensorflow as tf
from Reset_data import create_directory


def split_data(path: str) -> str:
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
    train_path_csv = glob(f"{dataset_path}/*.csv")[0]
    if flag:
        path = pd.read_csv(train_path_csv)

        real = path[path["label"] == 0].file_name
        ai = path[path["label"] == 1].file_name

        for i in real:
            shutil.copy(f"{dataset_path}/{i}", f"{dataset_path}/classes/real")

        for i in ai:
            shutil.copy(f"{dataset_path}/{i}", f"{dataset_path}/classes/ai")
    print("done")

    collect()


def preprocess_data(train_path: str) -> None:
    data = tf.keras.utils.image_dataset_from_directory(
        train_path, shuffle=True, seed=42, image_size=(224, 224)
    ).map(lambda x, y: (x / 255.0, y))

    size = len(data)

    train_size = round(size * 0.7)
    val_size = round(size * 0.2)

    test = data.skip(train_size + val_size).prefetch(tf.data.AUTOTUNE)
    # train = data.take(train_size).prefetch(tf.data.AUTOTUNE)
    # val = data.skip(train_size).take(val_size).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model('finished_model.keras')

    print(model.evaluate(test))
