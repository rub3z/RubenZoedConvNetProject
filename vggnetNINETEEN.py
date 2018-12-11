"""
Created on Mon Dec 10 22:29:26 2018

@author: Rub3z
"""
#VGGNet 19. The big one.
# Has a bare ImageDataGenerator for use on
# datasets that have already been augmented (or don't need it).
#Written by Ruben Baerga for CECS 456 Fall 2018
#Credit to Kacper https://www.kaggle.com/ammon1 
#for a generator method that catches exceptions 
#that may arise from corrupt data.
#plot_loss_accuracy function is from Keras_MNIST_HW.
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import LeakyReLU, BatchNormalization
import matplotlib.pyplot as plt
import json

from keras.preprocessing.image import ImageDataGenerator
print(keras.backend.image_data_format())

batch_size = 32
num_classes = 5

# input image dimensions
target_size = (128,128)

dataGenerator = ImageDataGenerator()

training = dataGenerator.flow_from_directory(
        'dataset_augmented/training_set/',
        target_size=target_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=True
        )

test = dataGenerator.flow_from_directory(
        'dataset_updated/validation_set/',
        target_size=target_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=True
        )

#This generator function takes in a generator and catches
# exceptions that come up from it.
def generatorChecker(imageGenerator):
    while True:
        try:
            x, y = next(imageGenerator)
            yield x, y
        except:
            pass

#I'm making my own VGGNet now
myVGGNet = Sequential()
myVGGNet.add(ZeroPadding2D((1,1), input_shape=(128,128,3)))
myVGGNet.add(Conv2D(64, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(64, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(128, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(128, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(256, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(256, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(256, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(512, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(512, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(512, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(512, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(512, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(512, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(512, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(ZeroPadding2D((1,1)))
myVGGNet.add(Conv2D(512, kernel_size=(3, 3)))
myVGGNet.add(BatchNormalization())
myVGGNet.add(LeakyReLU())
myVGGNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

myVGGNet.add(Flatten())
myVGGNet.add(Dense(4096))
myVGGNet.add(LeakyReLU())
myVGGNet.add(Dropout(0.5))
myVGGNet.add(Dense(4096))
myVGGNet.add(LeakyReLU())
myVGGNet.add(Dropout(0.5))

myVGGNet.add(Dense(num_classes, activation='softmax'))

myVGGNet.summary()

from keras.optimizers import SGD
myVGGNet.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=SGD(lr=0.001, decay=0.000001, momentum=0.9),
              metrics=['accuracy'])

epochs = 6
training_steps = 4826
validation_steps = 28

history = myVGGNet.fit_generator(
        generatorChecker(training),
        steps_per_epoch=training_steps,
        epochs=epochs,
        verbose=1,
        validation_data=generatorChecker(test),
        validation_steps=validation_steps
        )

#Dump the history info to a file.
with open('results/vggnetData1.json', 'w') as f:
    json.dump(history.history, f)

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
        plt.savefig('results/vggnetPlot1')
        
plot_loss_accuracy(history)

