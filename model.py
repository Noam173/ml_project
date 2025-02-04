from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from plot import plot_training as plt


def create_model(train, val):

    model = Sequential()
    
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(512,512,3)))
    
    model.add(MaxPooling2D())
    
    model.add(Conv2D(32, (3,3), activation='relu'))
    
    model.add(MaxPooling2D())
    
    model.add(Conv2D(16, (3,3), activation='relu'))
    
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    model.summary()

    hist = model.fit(train, epochs=20, 
                     validation_data=val, 
                     callbacks=[early_stopping])

    plt(hist)