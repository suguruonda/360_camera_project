import numpy as np
import glob
from PIL import Image
import cv2
import json
import math
import os
import time
import h5py

in_file = '/data2/suguru/datasets/360camera/butterfly/crop_pram_undistort_new.h5'
f_crop = h5py.File(in_file, 'r')

base_path = "/data2/suguru/datasets/360camera/butterfly/"
#result_base_path = "/mv_users/suguru/project/360camera/reults/correspondence/"
result_base_path = "/mv_users/suguru/project/360camera/reults/correspondence_undistort/"
sd_labels = ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']

threshold = 4

#original_folder_list = glob.glob(base_path + "original/*")
original_folder_list = glob.glob(base_path + "undistort/*")
original_folder_list.sort()

debug = False
start_num = 188
end_number = 213

for k, ele in enumerate(original_folder_list[start_num:end_number]):
    sample_index = start_num + k
    x_index,y_index = f_crop["crop/offset"][sample_index]
    width,height = f_crop["crop/img_size"][sample_index]

    name = os.path.basename(ele)
    if not(debug):
        #result_folder = base_path + "correspondence/" + name + "/"
        result_folder = base_path + "correspondence_undistort/" + name + "/"
    else:
        result_folder = result_base_path + "/" + os.path.basename(ele) + "/"
    try:
        os.makedirs(result_folder)
    except FileExistsError:
        pass

    print(name)
    time_sta_total = time.time()
    out_file = result_folder + "correspondence_coordinate.h5"
    f_h5py = h5py.File(out_file, 'w')
    for rgbimg_count in [0,20]:
        f_h5py.create_group(str(rgbimg_count))
        for c in sd_labels:
            time_sta = time.time()
            img_files = glob.glob(ele + "/scan_images/" + str(rgbimg_count) + "/" + c + "/*.JPG")
            img_files.sort()
            
            images_i = np.empty((height,width, len(img_files)//4))
            for i,elem in enumerate(img_files[0:18:2]):
                images_i[:,:,i] = np.array(Image.open(elem).convert('L'))[y_index:y_index + height,x_index:x_index + width]

            images_ = np.empty((height,width, len(img_files)//4))
            for i,elem in enumerate(img_files[1:18:2]):
                images_[:,:,i] = np.array(Image.open(elem).convert('L'))[y_index:y_index + height,x_index:x_index + width]

            x_bit = (images_-images_i>threshold).astype("int")
            x_bit_ignore = np.sum((np.absolute(images_-images_i) <= threshold).astype("int"), axis=2)

            images_i = np.empty((height,width, len(img_files)//4))
            for i,elem in enumerate(img_files[18:37:2]):
                images_i[:,:,i] = np.array(Image.open(elem).convert('L'))[y_index:y_index + height,x_index:x_index + width]

            images_ = np.empty((height,width, len(img_files)//4))
            for i,elem in enumerate(img_files[19:37:2]):
                images_[:,:,i] = np.array(Image.open(elem).convert('L'))[y_index:y_index + height,x_index:x_index + width]

            y_bit = (images_-images_i>threshold).astype("int")
            y_bit_ignore = np.sum((np.absolute(images_-images_i) <= threshold).astype("int"), axis=2)

            x_bit_ignore = (x_bit_ignore == 0).astype("int")
            y_bit_ignore = (y_bit_ignore == 0).astype("int")
            
            img_map = np.zeros((height,width, 2), dtype=int)
            for i in range(x_bit.shape[2]):
                img_map[:,:,0] += x_bit[:,:,i] * 2 ** i
                img_map[:,:,1] += y_bit[:,:,i] * 2 ** i
                
                #im = Image.fromarray(x_bit[:,:,i].astype(np.uint8)*255)
                #im.save(result_tif_path + "x_dif" + str(i) + ".tif")
                #im = Image.fromarray(y_bit[:,:,i].astype(np.uint8)*255)
                #im.save(result_tif_path + "y_dif" + str(i) + ".tif")

            x = (img_map[:,:,0]*x_bit_ignore)
            y = (img_map[:,:,1]*y_bit_ignore)

            if debug:    
                #im = Image.fromarray(x)
                #im.save(result_folder + "x_map" + str(rgbimg_count).zfill(2) + "_" + c +".tif")
                #im = Image.fromarray(y)
                #im.save(result_folder + "y_map" + str(rgbimg_count).zfill(2) + "_" + c +".tif")
                cv2.imwrite(result_folder + "x_map" + str(rgbimg_count).zfill(2) + "_" + c +".tif", x)
                cv2.imwrite(result_folder + "y_map" + str(rgbimg_count).zfill(2) + "_" + c +".tif", y)
            #f = open(result_folder + "map" + str(rgbimg_count).zfill(2) + "_" + c +'.txt','w')
            #img_points = np.zeros((im0.shape[0],im0.shape[1], 3), dtype=np.uint8)
            #x_range = np.trim_zeros(np.unique(x))
            #y_range = np.trim_zeros(np.unique(y))
            #x = x.astype(int)
            #y = y.astype(int)

            xy = ((y << 9) + x) * ((x>0)*(y>0))
            xy_range = np.trim_zeros(np.unique(xy))
            coordinate_list = []
            for count, i in enumerate(xy_range):
                region = np.argwhere(xy==i)
                point = np.average(region, axis=0)
                coordinate_list.append([point[1],point[0],i])
            f_h5py[str(rgbimg_count)].create_dataset(c, data=coordinate_list)
            """
            for i in x_range:
                for j in y_range:
                    time_sta = time.time()
                    print("\r"+"total(%d,%d), curent(%d,%d)" % (len(x_range),len(y_range),i,j),end = "")
                    region = np.argwhere((x==i) & (y==j))
                    if region.shape[0] != 0:
                        point = np.average(region, axis=0)
                        #h = int(math.atan((j%32+1)/(i%32+1))/np.pi*2*179)
                        #cv2.circle(img_points, (int(point[1]+0.5),int(point[0]+0.5)), 3, color=(h,255,255), thickness=-1)
                        f.write(str(point[0]) + ", " + str(point[1]) + ", " + str((j<<9) + i) + '\n')                
                    print(time.time()- time_sta)
            """
            #f.close()

            #im = cv2.cvtColor(img_points, cv2.COLOR_HSV2RGB)
            #im = Image.fromarray(im)
            #im.save(result_folder + "centerpoint_map" + str(rgbimg_count).zfill(2) + "_" + c +".png")
            
            img_color = np.ones((height,width, 3))
            x = x.astype(float)
            y = y.astype(float)
            img_color[:,:,0] = np.arctan(np.divide(y%32+1, x%32+1, out=np.zeros_like(x), where=x!=0))/np.pi*2*179
            img_color[:,:,1] = 255
            img_color[:,:,2] = np.where((x ==0) | (y ==0), 0, 255)
            #im = Image.fromarray(img_color[:,:,0])
            im = cv2.cvtColor(img_color.astype(np.uint8), cv2.COLOR_HSV2RGB)
            im = Image.fromarray(im)
            im.save(result_folder + "projector_map" + str(rgbimg_count).zfill(2) + "_" + c +".png")
            print(str(rgbimg_count) + ":" + c + " : ",end = "")
            print(str(int((time.time()- time_sta)//60)) + " min " + str(int((time.time()- time_sta)%60)) + " sec")
    f_h5py.close()
    print(name + " : " + str((height,width)) +" : total time" + " : ",end = "")
    print(str(int((time.time()- time_sta_total)//60)) + " min " + str(int((time.time()- time_sta_total)%60)) + " sec")