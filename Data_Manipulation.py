import shutil
from glob import glob
import tensorflow as tf
from Reset_data import *
from sklearn.metrics import confusion_matrix
from plot import *
from model import create_model as model


def split_data(path: str) -> str:
    dataset_path, flag = create_directory(path=path)
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


def preprocess_data(classes_path: str, model_path: str, exist: bool) -> None:
    data = tf.keras.utils.image_dataset_from_directory(
        classes_path, shuffle=True, seed=42, image_size=(224, 224), batch_size=16
    ).map(lambda x, y: (x / 255.0, y))

    size = len(data)

    train_size = round(size * 0.7)
    val_size = round(size * 0.2)

    if not exist:
        train = data.take(train_size).prefetch(tf.data.AUTOTUNE)
        val = data.skip(train_size).take(val_size).prefetch(tf.data.AUTOTUNE).cache()
        model(train=train, val=val)
    else:
        test = data.skip(train_size + val_size).prefetch(tf.data.AUTOTUNE)
        test = test.rebatch(len(test))
        print(con_matrix(dataset=test, model=model_path))


def con_matrix(dataset: tf.data.Dataset, model) -> None:
    model = tf.keras.models.load_model(model)
    pred = []
    labels = []
    for x, y in dataset:
        pred.extend(model.predict(x, verbose=0).round())
        labels.extend(y)

    labels = np.array(labels).flatten()
    pred = np.array(pred).flatten()

    matrix = confusion_matrix(labels, pred)
    plot_con_matrix(matrix=matrix)
