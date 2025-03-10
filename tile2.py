import numpy as np
import glob
from PIL import Image
import cv2
import os
sd_labels = ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

base_path = "/data2/suguru/datasets/360camera/butterfly/crop/*"
base_path2 = "/data2/suguru/datasets/360camera/butterfly/correspondence/"
original_folder_list = glob.glob(base_path)
original_folder_list.sort()
img_list = []
imgmask_list = []
imgcol_list = []
for count,val in enumerate(original_folder_list):
    species_number = count + 1
    if species_number == 60 or \
    species_number == 61 or \
    species_number == 62 or \
    species_number == 65 or \
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
    species_number == 93 or \
    species_number == 94 or \
    species_number%4 == 1 or \
    species_number%4 == 2 or \
    species_number%4 == 3:
        pass
    else:
        for i in sd_labels:

            if species_number == 66 or species_number == 68:
                arg = 0
            else:
                arg = 0
            images_folders = val + "/images/" + i + "/*.png"
            img_list2 = glob.glob(images_folders)
            img_list2.sort() 
            img = cv2.imread(img_list2[arg])
            
            mask_folders = val + "/mask/" + i + "/*.png"
            img_list_m = glob.glob(mask_folders)
            img_list_m.sort() 
            mask = cv2.imread(img_list_m[arg])
            name = os.path.basename(val)
            col_name = base_path2 + "/" + name + "/projector_map00_" + i + ".png"
            col = cv2.imread(col_name)
            #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            #ret, thresh = cv2.threshold(mask, 127, 255, 0)
            #contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #idx_max = max(enumerate(contours), key=lambda contour:cv2.contourArea(contour[1]))[0]
            #x, y, width, height = cv2.boundingRect(contours[idx_max])

            h,  w = img.shape[:2]
            
            if w > h:
                c = h

            else:
                c = w
            
            #img_out = img[y:y+w,x:x+w]
            resized = cv2.resize(img[0:c,0:c,:], (100,100), interpolation = cv2.INTER_AREA)
            img_list.append(resized)

            resized = cv2.resize(mask[0:c,0:c,:], (100,100), interpolation = cv2.INTER_AREA)
            imgmask_list.append(resized)
            resized = cv2.resize(col[0:c,0:c,:], (100,100), interpolation = cv2.INTER_AREA)
            imgcol_list.append(resized)

start = 0
row = 16
"""
im_tile = concat_tile([img_list[start:start+row],\
img_list[start+row:start+row*2],\
img_list[start+row*2:start+row*3],\
img_list[start+row*3:start+row*4],\
img_list[start+row*4:start+row*5],\
img_list[start+row*5:start+row*6],\
img_list[start+row*6:start+row*7],\
img_list[start+row*7:start+row*8],\
img_list[start+row*8:start+row*9],\
])
"""
im_tile = concat_tile([img_list[start:start+row],\
imgmask_list[start:start+row],\
imgcol_list[start:start+row],\
img_list[start+row:start+row*2],\
imgmask_list[start+row:start+row*2],\
imgcol_list[start+row:start+row*2],\
img_list[start+row*2:start+row*3],\
imgmask_list[start+row*2:start+row*3],\
imgcol_list[start+row*2:start+row*3],\
img_list[start+row*3:start+row*4],\
imgmask_list[start+row*3:start+row*4],\
imgcol_list[start+row*3:start+row*4],\
img_list[start+row*4:start+row*5],\
imgmask_list[start+row*4:start+row*5],\
imgcol_list[start+row*4:start+row*5],\
img_list[start+row*5:start+row*6],\
imgmask_list[start+row*5:start+row*6],\
imgcol_list[start+row*5:start+row*6]
#img_list[start+row*6:start+row*7],\
#imgmask_list[start+row*6:start+row*7],\
#imgcol_list[start+row*6:start+row*7],\
#img_list[start+row*7:start+row*8],\
#imgmask_list[start+row*7:start+row*8],\
#imgcol_list[start+row*7:start+row*8],\
#img_list[start+row*8:start+row*9],\
#imgmask_list[start+row*8:start+row*9],\
#mgcol_list[start+row*8:start+row*9]
])

cv2.imwrite('concat_tile2.jpg', im_tile)