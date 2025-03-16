import tensorflow as tf


def predict_image(img_dir: str) -> None:
    files = tf.data.Dataset.list_files(img_dir)

    def image(img_dir):
        img = tf.io.read_file(img_dir)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224)) / 255
        return img

    dataset = files.map(image).batch(1)

    model = tf.keras.models.load_model("finished_model.keras")

    pred = model.predict(dataset)

    print("the model's predictions:")
    [print("real" if pred.round()[i] == 1 else "ai") for i in range(len(files))]
