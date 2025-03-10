import numpy as np
import glob
from PIL import Image
import cv2
import json

base_path = "/multiview/datasets/360camera/test/monarch-001/correspondence/"
threshold = 20

for rgbimg_count in range(1,2):
    for c_num in range(1,9):
    #for c_num in range(1,2):
        f_name = "*" + str(rgbimg_count).zfill(3) + "_camera" + str(c_num) + "*.JPG"

        img_files = glob.glob(base_path + f_name)
        img_files.sort()

        im0 = np.array(Image.open(img_files[0]))
        images_x = np.empty((im0.shape[0],im0.shape[1], len(img_files)//4))
        for i,elem in enumerate(img_files[0:18:2]):
            images_x[:,:,i] = np.array(Image.open(elem).convert('L'))
            
        images_y = np.empty((im0.shape[0],im0.shape[1], len(img_files)//4))
        for i,elem in enumerate(img_files[18:37:2]):
            images_y[:,:,i] = np.array(Image.open(elem).convert('L'))

        images_xi = np.empty((im0.shape[0],im0.shape[1], len(img_files)//4))
        for i,elem in enumerate(img_files[1:18:2]):
            images_xi[:,:,i] = np.array(Image.open(elem).convert('L'))
            
        images_yi = np.empty((im0.shape[0],im0.shape[1], len(img_files)//4))
        for i,elem in enumerate(img_files[19:37:2]):
            images_yi[:,:,i] = np.array(Image.open(elem).convert('L'))

        x_bit = (images_x-images_xi>threshold).astype("int")
        y_bit = (images_y-images_yi>threshold).astype("int")

        x_bit_ignore = np.sum((np.absolute(images_x-images_xi) <= threshold).astype("int"), axis=2)
        y_bit_ignore = np.sum((np.absolute(images_y-images_yi) <= threshold).astype("int"), axis=2)
        x_bit_ignore = (x_bit_ignore == 0).astype("int")
        y_bit_ignore = (y_bit_ignore == 0).astype("int")

        img_map = np.empty((im0.shape[0],im0.shape[1], 2))
        for i in range(x_bit.shape[2]):
            img_map[:,:,0] += x_bit[:,:,i] * 2 ** i
            img_map[:,:,1] += y_bit[:,:,i] * 2 ** i
            
            #im = Image.fromarray(x_bit[:,:,i].astype(np.uint8)*255)
            #im.save("x_dif" + str(i) + ".tif")
            #im = Image.fromarray(y_bit[:,:,i].astype(np.uint8)*255)
            #im.save("y_dif" + str(i) + ".tif")

        x = (img_map[:,:,0]*x_bit_ignore)
        y = (img_map[:,:,1]*y_bit_ignore)         
        im = Image.fromarray(x)
        im.save("x_map" + str(rgbimg_count).zfill(3) + "_camera" + str(c_num) +".tif")
        im = Image.fromarray(y)
        im.save("y_map" + str(rgbimg_count).zfill(3) + "_camera" + str(c_num) +".tif")
        
        f = open("map" + str(rgbimg_count).zfill(3) + "_camera" + str(c_num) +'.txt','w')
        img_points = np.zeros((im0.shape[0],im0.shape[1], 3), dtype=np.uint8)
        x_range = np.trim_zeros(np.unique(x)).astype(int)
        y_range = np.trim_zeros(np.unique(y)).astype(int)
        x = x.astype(int)
        y = y.astype(int)
        for i in x_range:
            for j in y_range:
                print("\r"+"total(%d,%d), curent(%d,%d)" % (len(x_range),len(y_range),i,j),end = "")
                region = np.argwhere((x==i) & (y==j))
                if region.shape[0] != 0:
                    point = np.average(region, axis=0)
                    h = int((j%32+1)/(i%32+1)/np.pi*2*180)
                    cv2.circle(img_points, (int(point[1]+0.5),int(point[0]+0.5)), 3, color=(h,255,255), thickness=-1)
                    f.write(str(point[0]) + ", " + str(point[1]) + ", " + str((j<<9) + i) + '\n')                
        f.close()

        im = cv2.cvtColor(img_points, cv2.COLOR_HSV2RGB)
        im = Image.fromarray(im)
        im.save("centerpoint_map" + str(rgbimg_count).zfill(3) + "_camera" + str(c_num) +".png")
        
        img_color = np.ones((im0.shape[0],im0.shape[1], 3))
        print(str(c_num))
        x = x.astype(float)
        y = y.astype(float)
        img_color[:,:,0] = np.arctan(np.divide(y%32+1, x%32+1, out=np.zeros_like(x), where=x!=0))/np.pi*2*180
        img_color[:,:,1] = 255
        img_color[:,:,2] = np.where((x ==0) | (y ==0), 0, 255)
        #im = Image.fromarray(img_color[:,:,0])
        im = cv2.cvtColor(img_color.astype(np.uint8), cv2.COLOR_HSV2RGB)
        im = Image.fromarray(im)
        im.save("projector_map" + str(rgbimg_count).zfill(3) + "_camera" + str(c_num) +".png")
        