"""
A Python scripts for active learning
Using active learning approach to select data for K-iteration fine-tuning
Author: xinyi
Date: 20210824
"""
import sys
sys.path.insert(0, '/home/xinyi/Documents/myrobot')

import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import random
import shutil

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from tool.train_helper import *
from tool.data_generator import PickDataGenerator
from utils.image_proc_utils import *

class ActiveLearner():
    def __init__(self, pool, unlabeled_pool, labeled_pool, patch_ratio):
        """
        Arguments:
            pool {str} -- default dataset
            unlabeled_pool {str} -- default path to unlabeled pool
            labeled_pool {str} -- default path to labeled pool
            patch_ratio {float} -- patch selection ratio
        """
        self.pool = pool
        self.unlabeled_pool = unlabeled_pool
        self.labeled_pool = labeled_pool
        self.patch_ratio = patch_ratio
    
    def prepare_pools(self):
        # check if both pools are available
        # create both pools 
        for path in [self.unlabeled_pool, self.labeled_pool]:
            if os.path.isdir(path):
                if os.listdir(path)!=[]:
                    for f in os.listdir(path):
                        os.remove(os.path.join(path, f))
            else: 
                os.mkdir(path)
            
    def select_labeled_data(self):
        """Use our expert knowledge to select the necessary data
           Just for the first time training
        """
        # prepare both pools
        self.prepare_pools()
        ext='png'
        files = glob.glob(os.path.join(self.pool, "*.%s" % ext))
        for f in files:
            fname = f.split('/')[-1]
            pi, _, _, clutter, action, label = fname.split('_')
            success = label[0]
            if success == '1' and (clutter =='3' or clutter =='4') and (action =='4' or action=='5'or action=='6'):
                shutil.copyfile(f, os.path.join(self.unlabeled_pool, fname))
            elif success == '1' and clutter == '4' and (action == '0' or action == '1'):
                shutil.copyfile(f, os.path.join(self.unlabeled_pool, fname))
            elif success == '1' and clutter == '0' and action=='6':
                shutil.copyfile(f, os.path.join(self.unlabeled_pool, fname))
            elif success == '1' and (clutter == '2' or clutter == '1')and (action == '0' or action=='6'):
                shutil.copyfile(f, os.path.join(self.unlabeled_pool, fname))

            else:
                shutil.copyfile(f, os.path.join(self.labeled_pool, fname))
    
    def validate_data_with_query(self, model_path):
        """Load pre-trained model and check out the visualization result
           This function is optional
        Arguments:
            model_path {str} -- path to pre-trained model
        """

        sampled_actions = to_categorical(list(range(7)), 7)
        model = load_model(model_path) 

        model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics='accuracy')

        ext='png'
        files = glob.glob(os.path.join(self.unlabeled_pool, "*.%s" % ext))
        all_files = len(files)
        q = 0

        for i in range(all_files):
            f = files[i]
            fname = f.split('/')[-1]
            pi, u, v, clutter, action, tail = fname.split('_')
            img = cv2.resize(cv2.imread(f, 0), (224,224))
            img = adjust_grayscale(img)
            newu = float(224*int(u)/512)
            newv = float(224*int(v)/512)
            pose = [newu,newv]

            prob = []
            # print(f"==== img {i}====")
            for j in range(7):
                res=model.predict([np.array([img]),np.array([pose]), np.array([sampled_actions[j]])])
                prob.append(res[0][1])
                # print(res)

            # policy
            final_prob = 0
            prob = np.array(prob)
            if (prob < 0.5).all(): 
                char = 0
                final_prob = prob[6]
            elif (prob >= 0.5).all():
                char = 0
                final_prob = prob[0]
            else: 
                for k in range(7):
                    if prob[k] > 0.5:
                        char = k
                        final_prob = prob[k]
                        break

            # print("Best Action {}: {}".format(char, final_prob))
            # print("Labeled Action {}: {}".format(action, prob[int(action)]))

            query = 0 # 0:not logical data, don't training, 1: logical
            
            p = char # predict
            l = int(action) # labeled

            """LOGICAL"""
            if int(tail[0]): # positive examples
                if l >= p:
                    query = 1 # logical
            else: # negative examples
                if l < p: 
                    query = 1 #logical
            """UNLOGICAL"""
            # if int(tail[0]): # positive examples
            #     if l < p:
            #         query = 1 # unlogical
            # else: # negative examples
            #     if l >= p: 
            #         query = 1 # unlogical
                    
            print("Search for examples ... ")
            # if query: # with old label
            #     q += 1

            #     new_f = os.path.join(self.labeled_pool, fname)
            #     shutil.move(f, new_f)
            #     if q % 10: 
            #         print(f"Query {q} data ... ")

            if query: # revise old labels
                q += 1
                if prob[l] == 1: 
                    new_fname = '_'.join([pi, u, v, clutter, action,'1.png'])
                else:
                    new_fname = '_'.join([pi, u, v, clutter, action,'0.png'])
                new_f = os.path.join(self.labeled_pool, new_fname)
                shutil.move(f, new_f)
                if q % 10: 
                    print(f"Query {q} data ... ")
                    
            # if query: # with new label, always success
            #     q += 1
            #     new_fname = '_'.join([pi, u, v, clutter, str(char),'1.png'])
            #     new_f = os.path.join(self.labeled_pool, new_fname)
            #     shutil.move(f, new_f)
            #     if q % 10: 
            #         print(f"Query {q} data ... ")

            if q > int(all_files*self.patch_ratio):
                print("Totally transfer {} samples".format(q))
                break

if __name__ == '__main__':
    pool = '/home/xinyi/Documents/dataset/picking_reformatted/'
    labeled_pool = '/home/xinyi/Documents/dataset/labeled_pool/'
    unlabeled_pool = '/home/xinyi/Documents/dataset/unlabeled_pool/'
    model_path= '/home/xinyi/Documents/myrobot/learning/model/Logi_AL_20210901_131437.h5'

    al = ActiveLearner(pool,unlabeled_pool,labeled_pool,0.4)
    al.select_labeled_data()
    # al.validate_data_with_query(model_path)

    