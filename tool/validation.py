import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
np.set_printoptions(suppress=True)
import sys
sys.path.insert(0, './')
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from tool.data_generator import ASPDataGenerator

def predict(model, X):
    W = model.get_weights()

def validate_data(data_dir, model):
    # action_names = ['lift', 'half', 'full', 'fullspin', '2full']

    sdg = ASPDataGenerator(data_dir, 224)
    images, actions, poses, labels = sdg.load_data()
    num = len(labels)
    val_data,_ = sdg.split_data([images, actions, poses, labels], num)


    val_images, val_actions, val_poses, val_labels = val_data
    sampled_actions = to_categorical(list(range(7)), 7)


    # set up the figure
    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the images: each image is 28x28 pixels
    for i in range(50):
        ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
        # ax.imshow(val_images[i,:],cmap=plt.cm.gray_r, interpolation='nearest') # rgb

        ax.imshow(val_images[i],cmap=plt.cm.gray, interpolation='nearest') # grayscale

        ax.plot(val_poses[i][0],val_poses[i][1], 'o', color='y')
        prob = []
        print(f"==== img {i} ====")
        for j in range(7):
            res=model.predict([np.array([val_images[i]]),np.array([val_poses[i]]), np.array([sampled_actions[j]])])
            prob.append(res[0][1])

        final_prob = 0
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
        print("Best Action {}: {}".format(char, final_prob))
        print("Labeled Action {}: {}".format(val_actions[i], prob[int(val_actions[i])]))

        p = int(char)
        l = val_actions[i]
        query = 0 # 0:unlogical data, don't training, 1: logical
        if val_labels[i]: # positive examples
            if l >= p:
                query = 1 # logical
        else: # negative examples
            if l < p: 
                query = 1 # logical
        
        ax.text(0, 30, char, color='red',fontsize=15)
        ax.text(0, 55, str(val_actions[i])+"->"+str(val_labels[i]), color='green',fontsize=15)
        if query == 0: 
            ax.text(0, 70, 'not logical', color='red',fontsize=12)

    plt.show()

if __name__ == '__main__':
    val_data_dir = ""
    model_dir = ""

    model = load_model(model_dir)

    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics='accuracy')

    validate_data(val_data_dir,model)

    
  