#Used for connecting to plaidml, which allows for the usage of our integrated Macbook Pro GPU
#Comment the following 2 lines and the computations will execute in the CPU
import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
from matplotlib import pyplot as plotimport matplotlib.pyplot as plt


#Must first have a saved file that contains all our training set but in matrix form
#To do so, use the methos in data.py
x = np.load("dataSetInNumpyFormat/ImageSet-100.npy")
y = np.load("dataSetInNumpyFormat/LabelSet-100.npy")
xT = np.load("dataSetInNumpyFormat/ImageSet-Validation-100.npy")
yT = np.load("dataSetInNumpyFormat/labelSet-Validation-100.npy")

#Images are 100 x 100 x 3
print("Test Image Shape: ", x.shape)
print("Test Label Shape: ", y.shape)
print("Validation Image Shape: ", xT.shape)
print("Validation Label Shape: ", yT.shape)


# Normalize data set to 0-to-1 range
x = x.astype("float32")
x = x / 255
xT = xT.astype("float32")
xT = xT / 255

# Convert class vectors to binary class matrices
# We have 5 different classes or pictures, 
#1)Drawings, 2)Engraving, 3)Iconography, 4)Painting, 5)Sculpture
y = keras.utils.to_categorical(y, 5)
yT = keras.utils.to_categorical(yT, 5)


# Create a model and add layers
model = Sequential()

model.add(Conv2D ( filters = 32, kernel_size = (7, 7), strides=3, padding='valid', activation='relu', data_format='channels_last', input_shape = (100,100,3) ) )

model.add(MaxPooling2D( pool_size=(2,2), strides=2, padding='same' ) )

model.add(Dropout(0.3))

model.add(Conv2D( filters = 64, kernel_size = (5, 5), padding='same', strides=1, activation='relu' ) )

model.add(MaxPooling2D( pool_size=(2,2), strides=2, padding='same' ) )

model.add(Dropout(0.5))

model.add(Conv2D( filters = 128, kernel_size = (3, 3), padding='same', strides=1, activation='relu' ) )

model.add(Conv2D( filters = 128, kernel_size = (3, 3), padding='same', strides=1, activation='relu' ) )

model.add(Conv2D( filters = 64, kernel_size = (3, 3), padding='same', strides=1, activation='relu' ) )

model.add(MaxPooling2D( pool_size=(3,3), strides=2, padding='same' ) )

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense( 128, activation='relu' ) )

model.add(Dropout(0.5))

model.add(Dense( 128, activation='relu' ) )

model.add(Dense( 5, activation='softmax') )



# Compile the model
#Using stochastic gradient decent with the following parameters
sgd = optimizers.SGD(lr=0.01, decay=0.0005, momentum=0.9, nesterov=False)
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['mae', 'accuracy']
)
model.summary()


# Train the model
history = model.fit(
    x,
    y,
    batch_size=128,
    epochs=600,
    validation_data=(xT, yT),
    shuffle=True
)

#Show the scores we got from out trained model
score = model.evaluate(x, y, verbose=0)
print('\nScores')
print('\tTest loss:', score[0])
print('\tMean absolute error: ', score[1])
print('\tTest accuracy:', score[2])


#Show charts
#This definition is from: https://github.com/keras-team/keras/tree/master/examples
def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_acc"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)
    

plot_loss_accuracy(history)
















