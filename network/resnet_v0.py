import numpy as np
np.random.seed(123)
from math import *
import os
import matplotlib.pyplot as plt
import glob
import cv2
import time

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pickle as p
from tqdm import tqdm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAvgPool2D,AveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, Input, concatenate, add
from tensorflow.keras.optimizers import Adam, RMSprop

class ResNet(tf.keras.models.Sequential):
    def __init__(self, layer_num, input_shape, output_unit):
        self.inputs = Input(shape=input_shape)
        self.output_unit = output_unit
        self.stack_n = int((layer_num - 2) / 6)

    def residual_block(self, inputs, channels, strides=(1, 1)):
        net = BatchNormalization(momentum=0.9, epsilon=1e-5)(inputs)
        net = Activation('relu')(net)

        if strides == (1, 1):
            shortcut = inputs
        else:
            shortcut = Conv2D(channels, (1, 1), strides=strides)(net)

        net = Conv2D(channels, (3, 3), padding='same', strides=strides)(net)
        net = BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
        net = Activation('relu')(net)
        net = Conv2D(channels, (3, 3), padding='same')(net)

        net = add([net, shortcut])
        return net

    def build_res_net(self, inputs):
        net = Conv2D(16, (3, 3), padding='same')(inputs)

        for i in range(self.stack_n):
            net = self.residual_block(net, 16)

        net = self.residual_block(net, 32, strides=(2, 2))
        for i in range(self.stack_n - 1):
            net = self.residual_block(net, 32)

        net = self.residual_block(net, 64, strides=(2, 2))
        for i in range(self.stack_n - 1):
            net = self.residual_block(net, 64)

        net = BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
        net = Activation('relu')(net)
        net = AveragePooling2D(8, 8)(net)
        net = Flatten()(net)
        net = Dense(self.output_unit, activation='softmax')(net)
        return net

    def build_model(self):
        outputs = self.build_res_net(self.inputs)
        model = Model(self.inputs, outputs)
        return model

if __name__ == '__main__':
    # get model
    resnet = ResNet(50, (500, 500, 3), 2)
    model = resnet.build_model()
    model.summary()
