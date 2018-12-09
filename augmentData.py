# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:07:16 2018

@author: Rub3z
"""

#Made to create a whole augmented data set 10x larger than the old one.

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

batch_size = 1
num_classes = 5

# input image dimensions
target_size = (128,128)

dataGenerator = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.5,1.5),
        shear_range=0.2,
        zoom_range=(0.8,1),
        fill_mode='reflect',
        horizontal_flip=True,
        rescale = 1./255
        )

trainDrawings = dataGenerator.flow_from_directory(
        'dataset_updatedMod/training_set/drawings/',
        target_size=target_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,
        save_to_dir='dataset_augmented/training_set/drawings',
        save_prefix='aug',
        interpolation='lanczos'
        )

trainEngraving = dataGenerator.flow_from_directory(
        'dataset_updatedMod/training_set/engraving/',
        target_size=target_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,
        save_to_dir='dataset_augmented/training_set/engraving',
        save_prefix='aug',
        interpolation='lanczos'
        )

trainIconography = dataGenerator.flow_from_directory(
        'dataset_updatedMod/training_set/iconography/',
        target_size=target_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,
        save_to_dir='dataset_augmented/training_set/iconography',
        save_prefix='aug',
        interpolation='lanczos'
        )

trainPainting = dataGenerator.flow_from_directory(
        'dataset_updatedMod/training_set/painting/',
        target_size=target_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,
        save_to_dir='dataset_augmented/training_set/painting',
        save_prefix='aug',
        interpolation='lanczos'
        )

trainSculpture = dataGenerator.flow_from_directory(
        'dataset_updatedMod/training_set/sculpture/',
        target_size=target_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,
        save_to_dir='dataset_augmented/training_set/sculpture',
        save_prefix='aug',
        interpolation='lanczos'
        )


def genExcept(gen):
    try:
        gen.next()
    except:
        pass

for i in range(11070):
    genExcept(trainDrawings)

for i in range(7600):
    genExcept(trainEngraving)

for i in range(20790):
    genExcept(trainIconography)

for i in range(21280):
    genExcept(trainPainting)

for i in range(17450):
    genExcept(trainSculpture)



testDrawings = dataGenerator.flow_from_directory(
        'dataset_updatedMod/validation_set/drawings/',
        target_size=target_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,
        save_to_dir='dataset_augmented/validation_set/drawings',
        save_prefix='aug',
        interpolation='lanczos'
        )

testEngraving = dataGenerator.flow_from_directory(
        'dataset_updatedMod/validation_set/engraving/',
        target_size=target_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,
        save_to_dir='dataset_augmented/validation_set/engraving',
        save_prefix='aug',
        interpolation='lanczos'
        )

testIconography = dataGenerator.flow_from_directory(
        'dataset_updatedMod/validation_set/iconography/',
        target_size=target_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,
        save_to_dir='dataset_augmented/validation_set/iconography',
        save_prefix='aug',
        interpolation='lanczos'
        )

testPainting = dataGenerator.flow_from_directory(
        'dataset_updatedMod/validation_set/painting/',
        target_size=target_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,
        save_to_dir='dataset_augmented/validation_set/painting',
        save_prefix='aug',
        interpolation='lanczos'
        )

testSculpture = dataGenerator.flow_from_directory(
        'dataset_updatedMod/validation_set/sculpture/',
        target_size=target_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=False,
        save_to_dir='dataset_augmented/validation_set/sculpture',
        save_prefix='aug',
        interpolation='lanczos'
        )


def genExcept(gen):
    try:
        gen.next()
    except:
        pass

for i in range(1220):
    genExcept(testDrawings)

for i in range(840):
    genExcept(testEngraving)

for i in range(2310):
    genExcept(testIconography)

for i in range(2360):
    genExcept(testPainting)

for i in range(1930):
    genExcept(testSculpture)


