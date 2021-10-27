import numpy as np
np.random.seed(123)
from math import *
import os
import matplotlib.pyplot as plt
import glob
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2
import time
from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAvgPool2D, AveragePooling2D, BatchNormalization, Flatten, Dense, \
    Input, add, Activation
from tensorflow.keras.utils import plot_model

class AlexNet(tf.keras.models.Sequential):
    def __init__(self, input_shape, output_units):
        super().__init__()

        self.add(Conv2D(96, kernel_size=(11, 11), strides=4,
                        padding='valid', activation='relu',
                        input_shape=input_shape, kernel_initializer='he_normal'))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='valid', data_format=None))

        self.add(Conv2D(256, kernel_size=(5, 5), strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal'))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='valid', data_format=None))

        self.add(Conv2D(384, kernel_size=(3, 3), strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal'))

        self.add(Conv2D(384, kernel_size=(3, 3), strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal'))

        self.add(Conv2D(256, kernel_size=(3, 3), strides=1,
                        padding='same', activation='relu',
                        kernel_initializer='he_normal'))

        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='valid', data_format=None))

        self.add(Flatten())
        self.add(Dense(4096, activation='relu'))
        self.add(Dense(4096, activation='relu'))
        self.add(Dense(1000, activation='relu'))
        self.add(Dense(output_units, activation='softmax'))

        # self.compile(optimizer=tf.keras.optimizers.Adam(0.001),
        #              loss='categorical_crossentropy',
        #              metrics=['accuracy'])


if __name__ == '__main__':

    # get model
    model = AlexNet((227, 227, 3), 10)
    model.summary()



