import numpy as np
import glob
from PIL import Image
import cv2
import json
import math
import os
import time
import h5py
import matplotlib.pyplot as plt
import bundle_adjustment


base_path = "/data2/suguru/datasets/360camera/butterfly/"
result_base_path = "/mv_users/suguru/project/360camera/reults/correspondence/"
sd_labels = ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']

original_folder_list = glob.glob(base_path + "original/*")
original_folder_list.sort()

crop_file = '/data2/suguru/datasets/360camera/butterfly/crop_pram.h5'
f_crop = h5py.File(crop_file, 'r')

#cam_para_file = '/data2/suguru/datasets/360camera/camera_pram.h5'
#cam_para_file = '/data2/suguru/datasets/360camera/camera_pram_optimized.h5'
#cam_para_file = '/data2/suguru/datasets/360camera/camera_pram_2_no180.h5'
cam_para_file = '/data2/suguru/datasets/360camera/camera_pram_2.h5'
f_param = h5py.File(cam_para_file, 'r')

species_number = 7
#67
species_index = species_number - 1
name = os.path.basename(original_folder_list[species_index])

x,y = f_crop["crop/offset"][species_index]
w,h = f_crop["crop/img_size"][species_index]

cor_folder = base_path + "/correspondence/" + name + "/"
cor_file = cor_folder + "correspondence_coordinate.h5"
f_cor = h5py.File(cor_file, 'r')

cor_0_dic = {'camera1':None,'camera2':None,'camera3':None,'camera4':None,'camera5':None,'camera6':None,'camera7':None,'camera8':None}
cor_20_dic = {'camera1':None,'camera2':None,'camera3':None,'camera4':None,'camera5':None,'camera6':None,'camera7':None,'camera8':None}

undist_0_dic = {'camera1':None,'camera2':None,'camera3':None,'camera4':None,'camera5':None,'camera6':None,'camera7':None,'camera8':None}
undist_20_dic = {'camera1':None,'camera2':None,'camera3':None,'camera4':None,'camera5':None,'camera6':None,'camera7':None,'camera8':None}

#p_0_dic = {'camera1':None,'camera2':None,'camera3':None,'camera4':None,'camera5':None,'camera6':None,'camera7':None,'camera8':None}
#p_20_dic = {'camera1':None,'camera2':None,'camera3':None,'camera4':None,'camera5':None,'camera6':None,'camera7':None,'camera8':None}

p_list = [0 for i in range(16)]

num_0 = []
num_20 = []

for count, c in enumerate(sd_labels):
    mtx = f_param[c]['mtx'][:]
    dist = f_param[c]['dist'][:]

    cor_0_dic[c] = f_cor['0'][c][:]
    cor_20_dic[c] = f_cor['20'][c][:]






original_folder_list = glob.glob(base_path + "crop/*")
original_folder_list.sort()

name = os.path.basename(original_folder_list[species_index])

img1_path = original_folder_list[species_index] + "/images/" + "/" + sd_labels[1] + "/" + name[4:] + "-" + sd_labels[1] + "-" + "00" + ".png"

color = (0, 0, 255) 
img1 = cv2.imread(img1_path)
print(cor_0_dic['camera2'].shape)
for x,y,id in cor_0_dic['camera2']:
        img1 = cv2.circle(img1,(int(x),int(y)),2,color,1)
        #img1 = cv2.circle(img1,(int(x),int(y)),2,color,2)
cv2.imwrite("image_point_6.png", img1) 