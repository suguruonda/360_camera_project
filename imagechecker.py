import numpy as np
import glob
from PIL import Image
import cv2
import os
sd_labels = ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

base_path = "/data2/suguru/datasets/360camera/butterfly/original/*"
original_folder_list = glob.glob(base_path)
original_folder_list.sort()

for count,val in enumerate(original_folder_list):
    species_number = count + 1
    if species_number > 210:
        print(species_number)
        img_list = []
        imgs = []
        for i in sd_labels:
            images_folders = val + "/images/" + i + "/*.JPG"
            img_list2 = glob.glob(images_folders)
            img_list2.sort()
            img = cv2.imread(img_list2[0])
            resized = cv2.resize(img, (200,200), interpolation = cv2.INTER_AREA)
            imgs.append(resized)
        img_list.append(imgs)
        imgs = []
        for i in sd_labels:
            images_folders = val + "/images/" + i + "/*.JPG"
            img_list2 = glob.glob(images_folders)
            img_list2.sort()
            img = cv2.imread(img_list2[-1])
            resized = cv2.resize(img, (200,200), interpolation = cv2.INTER_AREA)
            imgs.append(resized)
        img_list.append(imgs)
        imgs = []
        for i in sd_labels:
            images_folders = val + "/scan_images/0/" + i + "/*.JPG"
            img_list2 = glob.glob(images_folders)
            img_list2.sort()
            img = cv2.imread(img_list2[0])
            resized = cv2.resize(img, (200,200), interpolation = cv2.INTER_AREA)
            imgs.append(resized)
        img_list.append(imgs)
        imgs = []
        for i in sd_labels:
            images_folders = val + "/scan_images/20/" + i + "/*.JPG"
            img_list2 = glob.glob(images_folders)
            img_list2.sort()
            img = cv2.imread(img_list2[0])
            resized = cv2.resize(img, (200,200), interpolation = cv2.INTER_AREA)
            imgs.append(resized)
        img_list.append(imgs)

        im_tile = concat_tile(img_list)

        cv2.imwrite('/data2/suguru/datasets/360camera/misc/butterfly/check/check_' + str(species_number) + '_.jpg', im_tile)