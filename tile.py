import numpy as np
import glob
from PIL import Image
import cv2
import os 

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

base_path = "/data2/suguru/datasets/360camera/butterfly/crop_undistort/*"

original_folder_list = glob.glob(base_path)
original_folder_list.sort()
img_list = []
for count,val in enumerate(original_folder_list):
    species_number = count + 1
    if species_number == 60 or \
    species_number == 61 or \
    species_number == 62 or \
    species_number == 64 or \
    species_number == 67 or \
    species_number == 69 or \
    species_number == 72 or \
    species_number == 73 or \
    species_number == 78 or \
    species_number == 79 or \
    species_number == 80 or \
    species_number == 83 or \
    species_number == 85 or \
    species_number == 87 or \
    species_number == 125 or \
    species_number == 130 or \
    species_number == 144 or \
    species_number == 160 or \
    species_number == 171 or \
    species_number == 175 or \
    species_number == 176 or \
    species_number == 179 or \
    species_number == 181 or \
    species_number == 182 or \
    species_number == 185 or \
    species_number == 187 or \
    species_number == 190 or \
    species_number == 194 or \
    species_number == 195 or \
    species_number == 197 or \
    species_number == 198 or \
    species_number == 201 or \
    species_number == 65 or \
    species_number == 93 or \
    species_number == 183 or \
    species_number == 199 or \
    species_number == 145 or \
    species_number == 94 or \
    species_number == 188 or \
    species_number == 189:
        pass
    else:
        if species_number == 66 or species_number == 68:
            arg = 30
        elif species_number == 111:
            arg = 20
        else:
            arg = 26
        images_folders = val + "/images/camera8/*.png"
        img_list2 = glob.glob(images_folders)
        img_list2.sort() 
        img = cv2.imread(img_list2[arg])
        
        mpath = "/data2/suguru/datasets/360camera/butterfly/crop_mask_undistort/" + os.path.basename(val)
        mask_folders = mpath + "/camera8/*.png"
        img_list_m = glob.glob(mask_folders)
        img_list_m.sort() 
        mask = cv2.imread(img_list_m[arg])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        idx_max = max(enumerate(contours), key=lambda contour:cv2.contourArea(contour[1]))[0]
        x, y, width, height = cv2.boundingRect(contours[idx_max])
        if width > height:
            w = width
            d = width - height
            d = d//2

        else:
            w = height
            d = height - width
            d = d//2
        img_out = img[y:y+w,x:x+w]
        resized = cv2.resize(img_out, (36,36), interpolation = cv2.INTER_AREA)
        img_list.append(resized)
        #print(species_number)

import copy
blank = copy.deepcopy(resized)
blank[:,:,:] = 255

for i in range(13):
    img_list.append(blank)

start = 0
row = 14

im_tile = concat_tile([img_list[start:start+row],\
img_list[start+row:start+row*2],\
img_list[start+row*2:start+row*3],\
img_list[start+row*3:start+row*4],\
img_list[start+row*4:start+row*5],\
img_list[start+row*5:start+row*6],\
img_list[start+row*6:start+row*7],\
img_list[start+row*7:start+row*8],\
img_list[start+row*8:start+row*9],\
img_list[start+row*9:start+row*10],\
img_list[start+row*10:start+row*11],\
img_list[start+row*11:start+row*12],\
img_list[start+row*12:start+row*13]
])
cv2.imwrite('concat_tile.jpg', im_tile)