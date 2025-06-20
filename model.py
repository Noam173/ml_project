import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Dropout,
)
from keras.models import Sequential
from keras.optimizers import Adam
from plot import plot_training as plt


def create_model(train: tf.data.Dataset, val: tf.data.Dataset) -> None:
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )

    lr_schedule = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
    )

    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D())

    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(Adam(5e-4), loss=tf.losses.binary_crossentropy, metrics=["accuracy"])

    hist = model.fit(
        train,
        epochs=30,
        validation_data=val,
        callbacks=[early_stopping, lr_schedule],
    )
    model.save("model.keras")
    print(plt(history=hist.history))
