import os
import matplotlib.pyplot as plt
import glob

data_dir = ""
ext='png'
files = glob.glob(os.path.join(data_dir, "*.%s" % ext))
clutters, actions, labels = [], [], []
for f in files:
    fname = os.path.split(f)[-1]
    pi, _, _, clutter, action, success = fname.split('_')
    clutters.append(int(clutter))
    actions.append(int(action))
    labels.append(int(success[0]))
print(len(files))

fig, axs = plt.subplots(1, 3,  figsize=(9, 3))
fig.suptitle('Labels, Clutters, and Actions')
axs[0].hist(labels, bins=[0,1,2], alpha=0.5)
axs[0].set_title('Success/Failure')

axs[1].hist(clutters, bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5], alpha=0.5)
axs[1].set_title('Clutters')
axs[2].hist(actions, bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6, 6.5], alpha=0.5)
axs[2].set_title('Actions')
fig.tight_layout()

plt.show()