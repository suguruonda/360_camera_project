import shutil
import glob
import os
sd_labels = ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']
"""
base_folder = "/data2/suguru/datasets/360camera/butterfly/crop_undistort/"

folder_list = glob.glob(base_folder + "/*")
for i in folder_list:
    f = i + "/*/*/*"
    f_list = glob.glob(f)
    for j in f_list:
        x = os.path.basename(j)
        os.rename(j, i + "/" + x)
    for k in sd_labels:
        os.rmdir(i + "/images/" + k)
    os.rmdir(i + "/images")
"""
base_folder = "/data2/suguru/datasets/360camera/butterfly/crop_mask_undistort/"

folder_list = glob.glob(base_folder + "/*")
for i in folder_list:
    f = i + "/*/*"
    f_list = glob.glob(f)
    for j in f_list:
        x = os.path.basename(j)
        os.rename(j, i + "/" + x)
    for k in sd_labels:
        os.rmdir(i + "/" + k)
