import tensorflow as tf
import matplotlib.pyplot as plt


def predict_image(img_dir: str) -> None:
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

    dataset = files.map(img_dir=image)

    model = tf.keras.models.load_model("model.keras")

    pred = model.predict(dataset)

    for i, file in enumerate(files.as_numpy_iterator()):
        file = file.decode("utf-8")
        plt.imshow(plt.imread(file))
        print("real" if pred.round()[i] == 1 else "ai")
        plt.show()
        print("=" * 300)
