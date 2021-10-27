"""
A Python scripts to train our model
A classification taken image+grasp+action as input, and output success/fail
Author: xinyi
Date: 20210819
---
Output: 
With data augmentation
Data: 700 -> 1047

"""
import numpy as np 
import pandas as pd
import os
import sys
sys.path.insert(0, '/home/xinyi/Documents/myrobot')
import glob
import cv2
import random
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from utils.base_utils import *
from utils.image_proc_utils import rotate_img
from tool.train_helper import positive_to_range_categorical
from tool.train_helper import negative_to_range_categorical
from tool.data_generator import PickDataGenerator

from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model

if __name__ == '__main__':

    data_dir = "/home/xinyi/Documents/dataset/picking_reformatted"
    sdg = PickDataGenerator(data_dir, resized_size=224)
    images, actions, poses, labels = sdg.load_data()

    num_after = len(labels)
    train_data, test_data = sdg.split_data([images, actions, poses, labels], num_after)

    result_print("Totally {} samples! ".format(num_after))
    
    
    train_images, train_actions, train_poses, train_labels = train_data
    test_images, test_actions, test_poses, test_labels = test_data
    train_num = train_images.shape[0]
    result_print("Totally {} training samples! ".format(train_num))

    train_actions = to_categorical(train_actions, 7)
    test_actions = to_categorical(test_actions, 7)

    train_labels = to_categorical(train_labels, 2)
    test_labels = to_categorical(test_labels, 2)

    train_images = np.array(train_images, dtype=np.float32)
    test_images = np.array(test_images, dtype=np.float32)
    train_poses = np.array(train_poses, dtype=np.float32)
    test_poses = np.array(test_poses, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.float32)

    # train_actions = np.array(train_actions/6, dtype=np.float32)
    # test_actions = np.array(test_actions/6, dtype=np.float32)
    
    # from network.sequential_mpnet import MPNetwork   
    # mpn = MPNetwork(action_input=7, grasp_input=2, image_shape= (256,256,3))
    # model = mpn.create_image_net()
    # model.summary()

    from network.aspnet import ASPNet
    aspnet= ASPNet(50, (224, 224, 1), 1024)
    model = aspnet.build_model()
    

    # model_dir = '/home/xinyi/Documents/myrobot/learning/model/Logi_AL_20210901_131437.h5'
    # model = load_model(model_dir)
    # model.summary()
    model.summary()
    model.compile(optimizer='adam',loss=binary_crossentropy,metrics=['accuracy'])

    #start training
    now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # batch_size = 15
    epochs = 30
    log_dir="logs/model_checkpoint_" + now_time
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    history = model.fit([train_images, train_poses, train_actions], train_labels, epochs=epochs,
                        callbacks=[tensorboard_callback],
                    # steps_per_epoch=train_num // batch_size,
                    validation_data=([test_images, test_poses, test_actions],test_labels))
    # history = model.fit([train_poses, train_images, train_actions], train_labels, epochs=epochs,
    #                     callbacks=[tensorboard_callback],
    #                 steps_per_epoch=train_num // batch_size,
    #                 validation_data=([test_poses, test_images, test_actions],test_labels))
    
    model.save('./model/Logi_AL_{}.h5'.format(now_time))