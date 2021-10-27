import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import optimizers, regularizers


class VGG16(tf.keras.models.Sequential):
    def __init__(self, input_shape, output_units, weight_decay):
        super().__init__()
        self.weight_decay = weight_decay
        self.add(Conv2D(64, (3, 3), activation='relu',
                        padding='same', input_shape=input_shape,
                        kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.add(MaxPooling2D((2, 2)))

        self.add(Conv2D(128, (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.add(Conv2D(128, (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.add(MaxPooling2D((2, 2)))

        self.add(Conv2D(256, (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.add(Conv2D(256, (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.add(Conv2D(256, (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.add(MaxPooling2D((2, 2)))

        self.add(Conv2D(512, (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.add(Conv2D(512, (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.add(Conv2D(512, (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.add(MaxPooling2D((2, 2)))

        self.add(Conv2D(512, (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.add(Conv2D(512, (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.add(Conv2D(512, (3, 3), activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.weight_decay)))

        self.add(Flatten())  # 2*2*512
        self.add(Dense(4096, activation='relu'))

        self.add(Dropout(0.5))
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(output_units, activation='softmax'))


if __name__ == '__main__':

    # get model
    model = VGG16((100, 100, 3), 1, weight_decay=5e-4)
    model.summary()
