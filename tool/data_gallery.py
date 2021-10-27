import os
import sys
sys.path.insert(0, './')
import cv2
import matplotlib.pyplot as plt
import glob
from grasp.graspability import Gripper, Graspability

def gallery_with_drawing(data_dir):
    ext='png'
    files = glob.glob(os.path.join(data_dir, "*.%s" % ext))
    all_files = len(files)
    gripper = Gripper()
    for f in files:
        fname = f.split('\\')[-1]
        pi, u, v, action, tail = fname.split('_')
        img = cv2.imread(fname)
        grasp =
        drawc, drawf = gripper.draw_grasp(
            best_grasp, drawf, drawc, left_margin, top_margin)

def draw_grasps(gripper, img_path, grasps, best_grasp):
    # draw grasps
    # drawc, _ = gripper.draw_grasps(
    #     grasps, img.copy(), im_cut.copy(), left_margin, top_margin, all=False)
    # _, drawf = gripper.draw_grasps(
    #     grasps, img.copy(), im_cut.copy(), left_margin, top_margin, all=True)
    img = cv2.imread(img_path)
    # cropped the necessary region (inside the bin)

    drawc, drawf = gripper.draw_uniform_grasps(
        grasps, img, im_cut, left_margin, top_margin)
    drawc, drawf = gripper.draw_grasp(
        best_grasp, drawf, drawc, left_margin, top_margin)
    return drawc

if __name__ == '__main__':
    exp_dir = 'D:\\material\\ICRA2022\\exp_results\\cp_short'
    gallery_with_drawing(exp_dir)