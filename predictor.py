import os
from typing_extensions import final

from numpy.lib.function_base import extract
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from re import X
import numpy as np 
import sys
sys.path.insert(0, '/home/xinyi/Workspace/myrobot')
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential

# from utils.base_utils import *

def grasp_policy(model, img, grasps):
    actions, probs = [], []
    for g in grasps:
        
        x, y = g[1], g[2]

        a, p = predict(model, img, x,y)
        actions.append(a)
        probs.append(p)
    print(actions, probs)
    action_indexes = sorted(range(len(actions)), key=lambda k: actions[k])
    return grasps[action_indexes[0]],actions[action_indexes[0]]


def predict_patch(model, img, grasps): 
    ch, cw, _ = img.shape
    
    g_num = len(grasps)
    pred_num = g_num * 7
    
    extract_poses = np.array(grasps, dtype=np.float64)[:,1:3]
    
    extract_poses[:,0] *= (224/cw)
    extract_poses[:,1] *= (224/ch)

    poses = np.repeat(extract_poses, 7, axis=0)

    sampled_actions = to_categorical(list(range(7)), 7)
    img = cv2.resize(img,(224,224))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    colorimg = cv2.cvtColor(np.tile(gray, (pred_num, 1)), cv2.COLOR_GRAY2BGR)
    images = np.reshape(colorimg, (pred_num, 224,224,3))

    actions = np.reshape(np.tile(sampled_actions, (g_num, 1)), (pred_num, 7))
    # poses = np.reshape(np.tile(extract_poses, (7, 1)), (pred_num, 2))
    # actions = to_categorical((np.repeat(list(range(7)),g_num)), 7)
    res = model.predict([images, poses, actions])

    alist, plist = [], []
    for i in range(0,pred_num, 7):
        # in the order of the input grasps
        prob = (res[i:(i+7)])[:,1]
        a, p = action_selection_policy(prob)
        alist.append(a)
        plist.append(p)

    # action_indexes = sorted(range(len(alist)), key=lambda k: alist[k])
    for i in range(7):
        if alist.count(i) != 0:
            final_a = i
            if alist.count(i) == 1:
                [index]=[j for j, x in enumerate(alist) if x == i]
                return grasps[index], final_a
            elif alist.count(6)==len(alist) and (np.array(plist) < 0.5).all()==True:
                return grasps[0], 6
            else:
                indexes = [j for j,x in enumerate(alist) if x == i]
                max_p = max([plist[j] for j in indexes])
                
                index=[j for j, x in enumerate(plist) if x == max_p]
                return grasps[index[0]], final_a
                
    # return grasps[action_indexes[0]],alist[action_indexes[0]]

def action_selection_policy(prob):
    final_prob =0
    if (prob < 0.5).all(): 
        # no prob is larger than 0.5 ==> Action 6
        char = 6
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
    return char, final_prob

def predict(model, img,x,y):
    # model_dir = "/home/xinyi/Workspace/myrobot/learning/model/Logi_AL_20210825_212307.h5"

    # model = load_model(model_dir)
    # model.compile(optimizer='adam',
    #                 loss='binary_crossentropy',
    #                 metrics='accuracy')
    ch, cw, _ = img.shape
    img = cv2.resize(img,(224,224))
    sampled_actions = to_categorical(list(range(7)), 7)
    
    newx = float(224*x/cw)
    newy = float(224*y/ch)

    prob = []
    print("+++++++++++++++++++++")
    for i in range(7):
        print([[newx,newy]])
        res=model.predict([np.array([img]), np.array([[newx,newy]]),np.array([sampled_actions[i]])],batch_size=1)
        prob.append(res[0][1])
        print(res)
    # sequential policy
    final_prob =0
    prob = np.array(prob)
    if (prob < 0.5).all(): 
        # no prob is larger than 0.5 ==> Action 6
        char = '6'
        final_prob = prob[6]
    elif (prob >= 0.5).all():
        char = '0'
        final_prob = prob[0]
    else: 
        for k in range(7):
            if prob[k] > 0.5:
                char = str(k)
                final_prob = prob[k]
                break
    # print("Best Action {}: {}".format(char, final_prob))
    return int(char), final_prob
class DPNet(Sequential):
    def __init__(self, input_shape):
        super().__init__()
        self.add(Conv2D(32, kernel_size=(9, 9), strides=(3, 3), activation='relu', input_shape=input_shape))
        self.add(Conv2D(16, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        self.add(Conv2D(8, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
        self.add(Conv2DTranspose(8, kernel_size=(3, 3), strides=(2, 2), activation='relu', output_padding=1))
        self.add(Conv2DTranspose(16, kernel_size=(5, 5), strides=(2, 2), activation='relu', output_padding=1))
        self.add(Conv2DTranspose(32, kernel_size=(9, 9), strides=(3, 3), activation='relu', output_padding=1))
        self.add(Conv2D(1, kernel_size=(2, 2), activation='relu'))

        self.compile(optimizer='adam',
            loss='mse',
            metrics=['mse'])

if __name__ == '__main__':
    img = cv2.imread('/home/xinyi/Workspace/myrobot/exp/20210825121327_299_502_0_.png')
    x =299
    y=502
    predict(img, x,y)