from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from plot import plot_training as plt
import tensorflow as tf


def create_model(train, val):

    early_stopping = EarlyStopping(monitor="val_loss",
                                   patience=5,         
                                   restore_best_weights=True,  
                                   verbose=1)

    
    model = Sequential()
    
    model.add(Conv2D(32, (3,3), 1, padding='same', activation='relu', input_shape=(128, 128, 3)))
    
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    
    model.add(BatchNormalization())

    model.add(MaxPooling2D())

    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))


    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    model.summary()

    hist = model.fit(train, epochs=20, 
                     validation_data=val, 
                     callbacks=[early_stopping])

    plt(hist.history)