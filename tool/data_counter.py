import os
import sys
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import random
import shutil

dir = '/home/xinyi/Documents/dataset/labeled_pool/'
ext='png'
files = glob.glob(os.path.join(dir, "*.%s" % ext))
clutters, actions, labels = [], [], []
for f in files:
    fname = f.split('/')[-1]
    pi, _, _, clutter, action, success = fname.split('_')
    clutters.append(int(clutter))
    actions.append(int(action))
    labels.append(int(success[0]))
print(len(files))

fig, axs = plt.subplots(2, 2)
fig.suptitle('Labels, Clutters, and Actions')
axs[0, 0].hist(labels, bins=[0,1,2], alpha=0.5)
axs[0, 0].set_title('Success/Failure')

axs[1, 0].hist(clutters, bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5], alpha=0.5)
axs[1, 0].set_title('Clutters')
axs[1, 1].hist(actions, bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6, 6.5], alpha=0.5)
axs[1, 1].set_title('Actions')
fig.tight_layout()


plt.show()