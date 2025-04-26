import tensorflow as tf
from plot import plot_images as plt


def predict_image(img_dir: str, model) -> None:
    files = tf.data.Dataset.list_files(img_dir)

    def image(img_dir: str) -> tf.Tensor:
        img = tf.io.read_file(img_dir)
        try:
            img = tf.image.decode_png(img, channels=3)
        except:
            img = tf.image.decode_bmp(img, channels=3)
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize(img, (224, 224)) / 255
        return img

    dataset = files.map(image)

    model = tf.keras.models.load_model(model)

    pred = model.predict(dataset)

    plt(pred=pred, files=files)
