from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from plot import plot_training as plt
import tensorflow as tf
from gc import collect

def create_model(train, val):

    early_stopping = EarlyStopping(monitor="val_loss",
                                   patience=5,         
                                   restore_best_weights=True,  
                                   verbose=1)

    tf.keras.backend.clear_session()
    
    model = Sequential()
    
    model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(224, 224, 3)))

    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    
    model.add(MaxPooling2D())

    model.add(Dropout(0.2))    
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    
    model.add(MaxPooling2D())

    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(224, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    
    collect()

    model.summary()

    hist = model.fit(train, epochs=10, 
                     validation_data=val, 
                     callbacks=[early_stopping])

    plt(hist.history)
