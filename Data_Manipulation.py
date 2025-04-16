import shutil
from glob import glob
import tensorflow as tf
from Reset_data import *
from sklearn.metrics import confusion_matrix
from model import create_model as model
from plot import *


def split_data(path: str) -> str:
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


def preprocess_data(classes_path: str, train: bool) -> None:
    data = tf.keras.utils.image_dataset_from_directory(
        classes_path, shuffle=True, seed=42, image_size=(224, 224), batch_size=16
    ).map(lambda x, y: (x / 255.0, y))

    size = len(data)

    train_size = round(size * 0.7)
    val_size = round(size * 0.2)

    if train:
        train = data.take(train_size).prefetch(tf.data.AUTOTUNE)
        val = data.skip(train_size).take(val_size).prefetch(tf.data.AUTOTUNE).cache()
        model(train=train, val=val)
    else:
        test = data.skip(train_size + val_size).prefetch(tf.data.AUTOTUNE)
        print(con_matrix(dataset=test))


def con_matrix(dataset: tf.data.Dataset) -> None:
    model = tf.keras.models.load_model("model.keras")
    pred = []
    labels = []
    for x, y in dataset.rebatch(len(dataset)):
        pred.extend(model.predict(x).round())
        labels.extend(y)

    labels = np.array(labels).flatten()
    pred = np.array(pred).flatten()

    matrix = confusion_matrix(labels, pred)
    plot_con_matrix(matrix)
