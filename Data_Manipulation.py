import shutil
from glob import glob
import tensorflow as tf
from Reset_data import create_directory
from plot import plot_con_matrix
import pandas as pd
import numpy as np
from model import create_model as model


def split_data(path: str) -> None:
    dataset_path, flag = create_directory(path=path)
    train_path_csv: str = glob(f"{dataset_path}/*.csv")[0]
    if flag:
        path: pd.DataFrame = pd.read_csv(train_path_csv)

        real: pd.Series = path[path["label"] == 0].file_name
        ai: pd.Series = path[path["label"] == 1].file_name

        for i in real:
            shutil.copy(dataset_path / i, dataset_path / "classes/real")

        for i in ai:
            shutil.copy(dataset_path / i, dataset_path / "classes/ai")
    print("done")


def preprocess_data(classes_path: str, model_path: str = None) -> None:
    data:tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        classes_path, shuffle=True, seed=42, image_size=(224, 224), batch_size=16
    ).map(lambda x, y: (x / 255.0, tf.cast(y, dtype=tf.int8)))

    size: int = len(data)

    train_size: int = round(size * 0.7)
    val_size: int = round(size * 0.2)

    if model_path:
        test:tf.data.Dataset = data.skip(train_size + val_size)
        test:tf.data.Dataset = test.rebatch(len(test)).prefetch(tf.data.AUTOTUNE)
        print(con_matrix(dataset=test, model=model_path))
    else:
        train:tf.data.Dataset = data.take(train_size).prefetch(tf.data.AUTOTUNE)
        val:tf.data.Dataset = data.skip(train_size).take(val_size).prefetch(tf.data.AUTOTUNE)
        model(train=train, val=val)


def con_matrix(dataset: tf.data.Dataset, model: str) -> None:
    model = tf.keras.models.load_model(model)
    pred: list[int] = []
    labels: list[int] = []
    for x, y in dataset:
        pred.extend(model.predict(x, verbose=0).round())
        labels.extend(y)

    labels = np.array(labels,dtype=np.int8).flatten()
    pred = np.array(pred,dtype=np.int8).flatten()

    matrix:tf.math = tf.math.confusion_matrix(labels, pred)
    plot_con_matrix(matrix=matrix)
