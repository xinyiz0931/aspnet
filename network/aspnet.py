import numpy as np
np.random.seed(123)
from math import *

from sklearn.preprocessing import MinMaxScaler
import pickle as p
from tqdm import tqdm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAvgPool2D,AveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, Input, concatenate, add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import plot_model, to_categorical

class ASPNet(Sequential):
    def __init__(self, layer_num, input_shape):
        self.inputs = Input(shape=input_shape)
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
        net = Dense(1024, activation='softmax')(net)
        return net

    def create_grasp_net(self, input_dim, regularizer=None):
        """Creates a simple two-layer MLP with inputs of the given dimension"""
        model = Sequential()
        model.add(Dense(256, input_dim=input_dim, activation="relu", kernel_regularizer=regularizer))
        # model.add(Dense(32, activation="relu", kernel_regularizer=regularizer))
        return model
        
    def create_action_net(self, input_dim, regularizer=None):
        """Creates a simple two-layer MLP with inputs of the given dimension"""
        model = Sequential()
        model.add(Dense(14, input_dim=input_dim, activation="relu", kernel_regularizer=regularizer))
        # model.add(Dense(128, input_dim=input_dim, activation="relu", kernel_regularizer=regularizer))
        # model.add(Dense(128, activation="relu", kernel_regularizer=regularizer))
        return model

    def build_model(self):
        resnet = self.build_res_net(self.inputs)
        gnet = self.create_grasp_net(2)
        anet = self.create_action_net(7)

        merge = concatenate([resnet, gnet.output, anet.output])
        
        x = Dense(256, activation="relu")(merge)
        outputs = Dense(2, activation="softmax")(x)

        model = Model([self.inputs, gnet.input, anet.input], outputs)
        return model

if __name__ == '__main__':
    # get model
    net= ASPNet(50, (224, 224, 3))
    model = net.build_model()
    model.summary()
    