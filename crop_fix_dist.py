import glob
import os
import numpy as np
import glob
import cv2
import h5py
import shutil
import time



sd_labels = ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']

base_folder = "/data2/suguru/datasets/360camera/butterfly/"
original_folder_list = glob.glob(base_folder + "undistort/*")
original_folder_list.sort()
out_file = '/data2/suguru/datasets/360camera/butterfly/crop_pram_undistort_new.h5'
f_h5py = h5py.File(out_file, 'r')

for k, ele in enumerate(original_folder_list):
    time_sta = time.time()
    name = os.path.basename(ele)

    dst_folder = base_folder + "/crop_undistort_new/" + name
    img_folder = dst_folder + "/images/" 
    #cor_folder = dst_folder + "/scan_images/"
    mask_folder = base_folder + "/crop_mask_undistort/" + name + '/'
    #cor_folder_0 = cor_folder + "/0/"
    #cor_folder_20 = cor_folder + "/20/"
    try:
        os.makedirs(dst_folder)
    except FileExistsError:
        pass
    try:
        os.makedirs(img_folder)
    except FileExistsError:
        pass
    """
    try:
        os.makedirs(cor_folder)
    except FileExistsError:
        pass
    try:
        os.makedirs(cor_folder_0)
    except FileExistsError:
        pass
    try:
        os.makedirs(cor_folder_20)
    except FileExistsError:
        pass
    """
    try:
        os.makedirs(mask_folder)
    except FileExistsError:
        pass
    
    print(name)
    x,y = f_h5py["crop/offset"][k]
    w,h = f_h5py["crop/img_size"][k]

    for count, i in enumerate(sd_labels):
        img_folder_c = img_folder + i +'/'
        try:
            os.makedirs(img_folder_c)
        except FileExistsError:
            pass
        """
        cor_folder_0_c = cor_folder_0 + i +'/'
        try:
            os.makedirs(cor_folder_0_c)
        except FileExistsError:
            pass

        cor_folder_20_c = cor_folder_20 + i +'/'
        try:
            os.makedirs(cor_folder_20_c)
        except FileExistsError:
            pass
        """
        mask_folder_c = mask_folder + i +'/'
        try:
            os.makedirs(mask_folder_c)
        except FileExistsError:
            pass
        
        f = ele + "/images/" + i + "/*.JPG"
        file_list = glob.glob(f)
        for j in file_list:
            img = cv2.imread(j)
            cv2.imwrite(img_folder_c + os.path.splitext(os.path.basename(j))[0] + ".png", img[y:y + h,x:x + w])
        """
        f = ele + "/scan_images/0/" + i + "/*.JPG"
        file_list = glob.glob(f)
        for j in file_list:
            img = cv2.imread(j)
            cv2.imwrite(cor_folder_0_c + os.path.splitext(os.path.basename(j))[0] + ".png", img[y:y + h,x:x + w])

        f = ele + "/scan_images/20/" + i + "/*.JPG"
        file_list = glob.glob(f)
        for j in file_list:
            img = cv2.imread(j)
            cv2.imwrite(cor_folder_20_c + os.path.splitext(os.path.basename(j))[0] + ".png", img[y:y + h,x:x + w])
        """
        f = base_folder + '/full_mask_undistort/' + name + '/' + i + "/*.png"
        file_list = glob.glob(f)
        for j in file_list:
            img = cv2.imread(j)
            cv2.imwrite(mask_folder_c + os.path.splitext(os.path.basename(j))[0] + ".png", img[y:y + h,x:x + w])
        

    print(str(int((time.time()- time_sta)//60)) + "min " + str(int((time.time()- time_sta)%60)) + "sec")
f_h5py.close()

