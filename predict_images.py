import cv2
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def predict_image(img_dir: str) -> None:
    if os.path.isdir(img_dir):
        img_list = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]
    else:
        img_list = [img_dir]

    model = tf.keras.models.load_model("finished_model.keras")

    for img_path in img_list:
        img = cv2.imread(img_path)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

        img = cv2.resize(img, (224, 224)) / 255
        img = img.reshape(1, 224, 224, 3)

        pred = model.predict(img)[0][0]
        print({"Real" if pred.round() == 1 else "ai"})

        print("=" * 188)
