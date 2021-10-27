import numpy as np 
import pandas as pd
import os
import sys
import glob
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

sys.path.insert(0, '/home/xinyi/Documents/myrobot')
from utils.image_proc_utils import *

from tensorflow.keras.utils import to_categorical, plot_model
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Input, Activation, Dropout, Lambda, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class PickDataGenerator():

    def __init__(self, data_dir, resized_size, data_num=None):
        self.data_dir = data_dir
        self.data_num = data_num
        self.resized_size = resized_size

    def generate_split_index(self, num, test_ratio=0.1):
        index_list = list(range(num))
        test_num = int(num*test_ratio)
        train_num = num - test_num
        test_index_list = random.sample(index_list, test_num)
        train_index_list = list(set(index_list)-set(test_index_list))
        return train_index_list, test_index_list
        
    def parse_dataset(self, ext='png'):   
        files = glob.glob(os.path.join(self.data_dir, "*.%s" % ext))
        if self.data_num is None:
            return random.sample(files, len(files))
        else:
            return random.sample(files, self.data_num)

    def load_positive_data(self,action_num=5):
        data_list = self.parse_dataset()
        images, actions, poses, labels = [],[],[],[]
        for data in data_list:
            filename = os.path.split(data)[1]
            filename = os.path.splitext(filename)[0]
            _, img_u, img_v, clutter, action, success = map(int, filename.split('_'))
            
            img = cv2.resize(cv2.imread(data, 0), (self.resized_size,self.resized_size))
            img = adjust_grayscale(img)
            newu = float(self.resized_size*img_u/512)
            newv = float(self.resized_size*img_v/512)
            pose = [newu,newv]

            if success:
                for i in range(action_num-action):
                    images.append(img)
                    actions.append(action+i)
                    poses.append(pose)
                    labels.append(success)
                    
        return images, actions, poses, labels
    

    def load_negative_data(self,action_num=5):
        data_list = self.parse_dataset()
        images, actions, poses, labels = [],[],[],[]
        for data in data_list:
            filename = os.path.split(data)[1]
            filename = os.path.splitext(filename)[0]
            _, img_u, img_v, clutter, action, success = map(int, filename.split('_'))
            
            img = cv2.resize(cv2.imread(data, 0), (self.resized_size,self.resized_size))
            img = adjust_grayscale(img)
            newu = float(self.resized_size*img_u/512)
            newv = float(self.resized_size*img_v/512)
            pose = [newu,newv]

            if success==0:
                for i in range(action+1):
                    """rotate the image to get four images"""
                    # original image
                    images.append(img)
                    actions.append(i)
                    poses.append(pose)
                    labels.append(success)
                    # rotated to get another three images
                    for j in [90,180,270]:
                        roimg = rotate_img(img,j)
                        ropose = rotate_point(img, pose, j)
                        images.append(roimg)
                        actions.append(action)
                        poses.append(ropose)
                        labels.append(success)

        return images, actions, poses, labels

    def load_data(self):

        data_list = self.parse_dataset()
        images, actions, poses, labels = [],[],[],[]

        for data in data_list:
            filename = os.path.split(data)[1]
            filename = os.path.splitext(filename)[0]
            _, img_u, img_v, clutter, action, success = map(int, filename.split('_'))
            # resize
            img = cv2.resize(cv2.imread(data,0), (self.resized_size,self.resized_size))
            img = adjust_grayscale(img)
            newu = float(self.resized_size*img_u/512)
            newv = float(self.resized_size*img_v/512)
            pose = [newu,newv]
            images.append(img)
            actions.append(action)
            poses.append(pose)
            labels.append(success)
        return images, actions, poses, labels

    def augment_all_data(self, action_num=7):
        """augment data by generating more labels, and rotating negative data"""
        data_list = self.parse_dataset()
        images, actions, poses, labels = [],[],[],[]
        

        for data in data_list:
            filename = os.path.split(data)[1]
            filename = os.path.splitext(filename)[0]
            _, img_u, img_v, clutter, action, success = map(int, filename.split('_'))

            img = cv2.resize(cv2.imread(data,0), (self.resized_size,self.resized_size))
            img = adjust_grayscale(img)
            newu = float(self.resized_size*img_u/512)
            newv = float(self.resized_size*img_v/512)
            pose = [newu,newv]

            if success:
                for i in range(action_num-action):
                    images.append(img)
                    actions.append(action+i)
                    poses.append(pose)
                    labels.append(success)
            else:
                for i in range(action+1):
                    """rotate the image to get four images"""
                    # original image
                    images.append(img)
                    actions.append(i)
                    poses.append(pose)
                    labels.append(success)
                    # rotated to get another three images
                    for j in [90,180,270]:
                        roimg = rotate_img(img,j)
                        ropose = rotate_point(img, pose, j)
                        images.append(roimg)
                        actions.append(action)
                        poses.append(ropose)
                        labels.append(success)

        return images, actions, poses, labels
    
    def augment_negative_data(self, action_num=7):
        """augment data by generating more labels, and rotating negative data"""
        data_list = self.parse_dataset()
        images, actions, poses, labels = [],[],[],[]
        

        for data in data_list:
            filename = os.path.split(data)[1]
            filename = os.path.splitext(filename)[0]
            _, img_u, img_v, clutter, action, success = map(int, filename.split('_'))

            img = cv2.resize(cv2.imread(data,0), (self.resized_size,self.resized_size))
            img = adjust_grayscale(img)
            newu = float(self.resized_size*img_u/512)
            newv = float(self.resized_size*img_v/512)
            pose = [newu,newv]

            if success:
                images.append(img)
                actions.append(action)
                poses.append(pose)
                labels.append(success)
            else:
                """rotate the image to get four images"""
                # original image
                images.append(img)
                actions.append(action)
                poses.append(pose)
                labels.append(success)

                roimg = rotate_img(img,180)
                ropose = rotate_point(img, pose, 180)
                images.append(roimg)
                actions.append(action)
                poses.append(ropose)
                labels.append(success)

                # rotated to get another three images
                # for j in [90,180,270]:
                #     roimg = rotate_img(img,j)
                #     ropose = rotate_point(img, pose, j)
                #     images.append(roimg)
                #     actions.append(action)
                #     poses.append(ropose)
                #     labels.append(success)
                

        return images, actions, poses, labels
    
    def augment_data(self, action_num=7):

        data_list = self.parse_dataset()
        images, actions, poses, labels = [],[],[],[]

        for data in data_list:
            filename = os.path.split(data)[1]
            filename = os.path.splitext(filename)[0]

            _, img_u, img_v, clutter, action, success = map(int, filename.split('_'))

            img = cv2.resize(cv2.imread(data, 0), (self.resized_size,self.resized_size))
            img = adjust_grayscale(img)
            newu = float(self.resized_size*img_u/512)
            newv = float(self.resized_size*img_v/512)
            pose = [newu,newv]

            if success:
                for i in range(action_num-action):
                    images.append(img)
                    actions.append(action+i)
                    poses.append(pose)
                    labels.append(success)
            else:
                for i in range(action+1):
                    images.append(img)
                    actions.append(i)
                    poses.append(pose)
                    labels.append(success)

        return images, actions, poses, labels

    def split_data(self, data, num, test_ratio=0.15):
        """Input must share the same number"""
        train_index, test_index = self.generate_split_index(num)
        train_data, test_data = [], []
        for each in data:
            tr = [each[i] for i in train_index]
            te = [each[i] for i in test_index]
            train_data.append(np.asarray(tr))
            test_data.append(np.asarray(te))
 
        return train_data, test_data
    
    def draw_data(self, images, poses):
        """Draw 50 images with grasp point for verification"""
        fig = plt.figure(figsize=(15, 7))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

        for i in range(50):
            ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
            ax.imshow(images[i,:],cmap=plt.cm.gray_r, interpolation='nearest')
            ax.plot(poses[i][0],poses[i][1], 'o', color='y')
        plt.show()

if __name__ == '__main__':
    
    data_dir = "/home/xinyi/Documents/dataset/labeled_pool"
    sdg = PickDataGenerator(data_dir, 224)

    images, actions, poses, labels = sdg.load_positive_data()
    positive_num = len(labels)
    print(f"Loaded {positive_num} positive samples! ")

    images, actions, poses, labels = sdg.load_negative_data()
    negative_num = len(labels)
    print(f"Loaded {negative_num} negative samples! ")

    # img = cv2.imread("/home/xinyi/Documents/dataset/picking_700/20210818113336_199_420_6_1.png",0)
    # img90 = rotate_img(img,90)
    # img180 = rotate_img(img,180)
    # img270 = rotate_img(img,270)
    # cv2.imshow("NoRotation", img)
    # cv2.imshow("Rotate90", img90)
    # cv2.imshow("Rotate180", img180)
    # cv2.imshow("Rotate270", img270)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # train_data,test_data = sdg.split_data([images, actions, poses, labels], num)

    # train_images, train_actions, train_poses, train_labels = train_data
    # test_images, test_actions, test_poses, test_labels = test_data

    # sdg.draw_data(train_images, train_poses)
    # train_data, test_data = sdg.data_split([images, actions, poses, labels], num_after)

    # train_images, train_actions, train_poses, train_labels = train_data
    # test_images, test_actions, test_poses, test_labels = test_data

    # print(np.count_nonzero(train_labels==1))
    # print(np.count_nonzero(train_labels==0))
