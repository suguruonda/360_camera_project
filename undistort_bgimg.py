import numpy as np
import cv2 as cv
import glob
import screeninfo
import os
import h5py
import time
sd_labels = ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']

base_folder = "/data2/suguru/datasets/360camera/"
original_folder = base_folder + "bg_images/"
#original_folder_list.sort()

dis_base_folder = base_folder + "bg_images_undistort/"

out_file = '/data2/suguru/datasets/360camera/camera_pram.h5'
f = h5py.File(out_file, 'r')


for i in sd_labels:
    dist_folder_c = dis_base_folder + i + "/"
    try:
        os.makedirs(dist_folder_c)
    except FileExistsError:
        pass
    mtx = np.array(f[i]['mtx'], dtype=np.float64)
    dist = np.array(f[i]['dist'], dtype=np.float64)
    img_folder_c = original_folder + i + "/*"
    img_list = glob.glob(img_folder_c)
    for j in img_list:
        img = cv.imread(j)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        #x, y, w, h = roi
        #dst = dst[y:y+h, x:x+w]
        cv.imwrite(dist_folder_c + os.path.basename(j), dst)

f.close()
    