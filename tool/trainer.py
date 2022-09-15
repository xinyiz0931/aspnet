"""
A Python scripts to train our model
A classification taken image+grasp+action as input, and output success/fail
Author: xinyi
Date: 20210819
"""
import numpy as np 
import os
import sys
sys.path.insert(0, './')
import datetime
import pandas as pd

from network.aspnet import ASPNet

from tool.data_generator import ASPDataGenerator

from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model

if __name__ == '__main__':

    data_dir = "D:\\Dataset\\aspnet_data\\picking_700_resized_reformatted"
    sdg = ASPDataGenerator(data_dir, resized_size=224)
    images, actions, poses, labels = sdg.load_data()

    num_after = len(labels)
    train_data, val_data = sdg.split_data([images, actions, poses, labels], num_after)
    
    train_images, train_actions, train_poses, train_labels = train_data
    val_images, val_actions, val_poses, val_labels = val_data
    train_num = train_images.shape[0]
    print(f"[*] Totally {num_after} samples! Train: {train_num}, validate: {num_after-train_num}")

    train_actions = to_categorical(train_actions, 7)
    val_actions = to_categorical(val_actions, 7)

    train_labels = to_categorical(train_labels, 2)
    val_labels = to_categorical(val_labels, 2)

    train_images = np.array(train_images, dtype=np.float32)
    val_images = np.array(val_images, dtype=np.float32)
    train_poses = np.array(train_poses, dtype=np.float32)
    val_poses = np.array(val_poses, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.float32)
    val_labels = np.array(val_labels, dtype=np.float32)

    aspnet= ASPNet(50, (224, 224, 1), 1024)
    model = aspnet.build_model()
    model.summary()
    
    model.compile(optimizer='adam',loss=binary_crossentropy,metrics=['accuracy'])

    #start training
    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_size = 15
    epochs = 30
    log_dir="./logs/model_checkpoint_" + now_time
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    history = model.fit([train_images, train_poses, train_actions], train_labels, epochs=epochs,
                        callbacks=[tensorboard_callback],
                        steps_per_epoch=train_num // batch_size,
                        validation_data=([val_images, val_poses, val_actions],val_labels))

    model.save('./model/model_{}.h5'.format(now_time))