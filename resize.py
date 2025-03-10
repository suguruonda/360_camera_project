import shutil
import glob
import os
import cv2

factor = 8
"""
base_folder = "/data2/suguru/datasets/360camera/butterfly/crop_undistort/"
dst_folder = "/data2/suguru/datasets/360camera/butterfly/crop_undistort_" + str(factor) + "/"
folder_list = glob.glob(base_folder + "/*")

for i in folder_list:
    foldername = os.path.basename(i)
    dest_path = dst_folder + foldername
    try:
        os.makedirs(dest_path)
    except FileExistsError:
        pass

    f = i + "/*"
    f_list = glob.glob(f)
    H_original, W_original = cv2.imread(f_list[0]).shape[:2]
    H = H_original//factor
    W = W_original//factor
    for j in f_list:
        x = os.path.basename(j)
        image_original = cv2.imread(j)
        resize_img = cv2.resize(image_original, (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dest_path + "/" + x, resize_img)
"""

base_folder = "/data2/suguru/datasets/360camera/butterfly/crop_mask_undistort/"
dst_folder = "/data2/suguru/datasets/360camera/butterfly/crop_mask_undistort" + str(factor) + "/"
folder_list = glob.glob(base_folder + "/*")

for i in folder_list:
    foldername = os.path.basename(i)
    dest_path = dst_folder + foldername
    try:
        os.makedirs(dest_path)
    except FileExistsError:
        pass

    f = i + "/*"
    f_list = glob.glob(f)
    H_original, W_original = cv2.imread(f_list[0]).shape[:2]
    H = H_original//factor
    W = W_original//factor
    for j in f_list:
        x = os.path.basename(j)
        image_original = cv2.imread(j)
        resize_img = cv2.resize(image_original, (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(dest_path + "/" + x, resize_img)


