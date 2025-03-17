import tensorflow as tf
import matplotlib.pyplot as plt


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
    
    real=False
    ai=False
    for i, file in enumerate(files.as_numpy_iterator()):
            
        if pred[i]>0.8 and real==False:
            real=True
            plt.imshow(plt.imread(file.decode("utf-8")))
            print('real')
            plt.show()
                
        if pred[i]<0.2 and ai==False:
            ai=True
            plt.imshow(plt.imread(file.decode("utf-8")))
            print('ai')
            plt.show()
            
        print(f'{file} is:')
        print('real' if pred.round()[i] == 1 else 'ai')
                
        