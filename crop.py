import glob
import os
import numpy as np
import glob
from PIL import Image
import cv2
import h5py
import time

make_crop_images = True

names = ["Papilio_Rutulus","Papilio_Zelicaon","Papilio_Indra","Papilio_Machaon","Parnassius_Smintheus",
        "Pieris_Rapae","Pontia_Protodice","Pontia_Occidentalis","Neophasia_Menapia","Euchloe_Ausonides",
        "Anthocharis_Julia","Colias_Philodice","Colias_Eurytheme","Colias_Christina","Celastrina_Ladon",
        "Plebejus_Melissa","Glaucopsyche_Lygdamus","Cupido_Amyntula","Plebejus_Acmon","Tharsalea_Rubidus",
        "Tharsalea_Heteronea","Lycaena_Hyllus","Tharsalea_Helloides","Polygonia_Satyrus","Nymphalis_Californica",
        "Hypaurotis_Crysalus","Strymon_Melinus","Satyrium_Sylvinus","Satyrium_Behrii","Callophrys_Gryneus",
        "Callophrys_Eryphon","Apodemia_Mormo","Nymphalis_Antiopa","Aglais_Milberti","Epargyreus_Clarus",
        "Thorybes_Pylades","Erynnis_Telemachus","Burnsius_Communis","Amblyscirtes_Vialis","Hesperia_Juba",
        "Ochlodes_Sylvanoides","Lon_Taxiles","Polites_Sabuleti","Vanessa_Atalanta","Vanessa_Cardui",
        "Junonia_Coenia","Chlosyne_Leanira","Chlosyne_Acastus","Euphydryas_Anicia","Phyciodes_Cocyta",
        "Phyciodes_Pulchella","Boloria_Kriemhild","Boloria_Selene","Adelpha_Eulalia","Limenitis_Weidemeyerii",
        "Limenitis_Archippus","Speyeria_Cybele","Speyeria_Coronis","Cercyonis_Pegala","Cercyonis_Pegala",
        "Papilio_Indra","Papilio_Zelicaon","Papilio_Polyxenes","Papilio_Machaon","Papilio_Machaon",
        "Papilio_Multicaudata","Papilio_Rutulus","Papilio_Eurymedon","Parnassius_Smintheus","Parnassius_Clodius",
        "Colias_Occidentalis","Colias_Eurytheme","Colias_Philodice","Colias_Alexandra","Colias_Meadii",
        "Colias_Scudderii","Nathalis_Iole","Neophasia_Menapia","Pontia_Occidentalis","Pontia_Protodice",
        "Pontia_Sisymbrii","Pontia_Beckerii","Pieris_Rapae","Pieris_Marginalis","Euchloe_Ausonides",
        "Euchloe_Lotta","Anthocharis_Julia","Anthocharis_Thoosa","Anthocharis_Cethura","Cercyonis_Sthenele",
        "Cercyonis_Oetus","Cercyonis_Meadii","Cercyonis_Pegala","Cercyonis_Pegala","Erebia_Epipsodea",
        "Erebia_Callias","Erebia_Magdalena","Coenonympha_California","Cyllopsis_Pertepida","Danaus_Gilippus",
        "Oeneis_Chryxus","Oeneis_Jutta","Oeneis_Bore","Oeneis_Melissa","Oeneis_Uhleri",
        "Neominois_Ridingsii"]

sd_labels = ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']

mtx_x_list = []
mtx_y_list = []

out_file = '/data2/suguru/datasets/360camera/camera_pram_2.h5'
#f_h5py = h5py.File(out_file, 'r+')
f_h5py = h5py.File(out_file, 'r')
for i in sd_labels:
    mtx_x_list.append(f_h5py[i + "/mtx"][0, 2])
    mtx_y_list.append(f_h5py[i + "/mtx"][1, 2])
f_h5py.close()
offset = []
crop_img_size = []

mtx_x_min = int(min(mtx_x_list))
mtx_y_min = int(min(mtx_y_list))
mtx_x_max = int(max(mtx_x_list)+1)
mtx_y_max = int(max(mtx_y_list)+1)

bg_base = "/data2/suguru/datasets/360camera/bg_images/"
base_folder = "/data2/suguru/datasets/360camera/butterfly/"
bg_names = ("_bg_f20e-1.JPG", "_bg_f25e-1.JPG", "_bg_f32e-1.JPG")
result_path = "/mv_users/suguru/project/360camera/crop_test/"

original_folder_list = glob.glob(base_folder + "original/*")
original_folder_list.sort()
threshold = 31

th_dic = {'camera1':{"low_H":[49,55,56],"low_S":[0,0,0],"low_V":[117,60,77],"high_H":[72,72,72],"high_S":[255,255,255],"high_V":[255,255,255]},\
          'camera2':{"low_H":[49,56,56],"low_S":[0,0,0],"low_V":[117,91,81],"high_H":[72,72,72],"high_S":[255,255,255],"high_V":[255,255,255]},\
          'camera3':{"low_H":[49,50,56],"low_S":[0,0,0],"low_V":[117,105,80],"high_H":[72,72,72],"high_S":[255,255,255],"high_V":[255,255,255]},\
          'camera4':{"low_H":[49,56,56],"low_S":[0,0,0],"low_V":[117,96,80],"high_H":[72,72,72],"high_S":[255,255,255],"high_V":[255,255,255]},\
          'camera5':{"low_H":[49,49,56],"low_S":[0,0,0],"low_V":[80,71,60],"high_H":[72,72,72],"high_S":[255,255,255],"high_V":[255,255,255]},\
          'camera6':{"low_H":[49,56,59],"low_S":[0,0,0],"low_V":[117,85,65],"high_H":[72,72,72],"high_S":[255,255,255],"high_V":[255,255,255]},\
          'camera7':{"low_H":[49,56,56],"low_S":[0,0,0],"low_V":[117,71,60],"high_H":[72,72,72],"high_S":[255,255,255],"high_V":[255,255,255]},\
          'camera8':{"low_H":[49,57,56],"low_S":[0,0,0],"low_V":[117,83,70],"high_H":[72,72,72],"high_S":[255,255,255],"high_V":[255,255,255]},}


"""
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
attempts=10
"""
#for k, ele in enumerate(names):
for k, ele in [(14,names[14]),(71,names[71])]:
    time_sta = time.time()
    species_number = k + 1
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
    species_number == 87:
        id = 2
    elif species_number == 65 or \
    species_number == 93:
        id = 3
    elif species_number == 94:
        id = 4
    else:
        id = 1
    name = ele + '-' + str(id).zfill(3)

    full_dst_folder = base_folder + "/full_mask/" + str(species_number).zfill(3) + "-" + name

    #Sprint(k)
    print(str(species_number) + ' ' + name)

    try:
        os.makedirs(full_dst_folder)
    except FileExistsError:
        pass

    x_min_list = []
    y_min_list = []
    x_max_list = []
    y_max_list = []

    if species_number in [7,8,9,10,11,12,76,77,78,79,80,81,82,83,84,85,86,87,88,89]:
        bg_arg = 2
    elif species_number in [6]:
        bg_arg = 1
    else:
        bg_arg = 0

    for count, i in enumerate(sd_labels):

        full_folder_c = full_dst_folder + '/' + i +'/'
        try:
            os.makedirs(full_folder_c)
        except FileExistsError:
            pass

        f = original_folder_list[species_number - 1] + "/images/" + i + "/*.JPG"
        file_list = glob.glob(f)
        file_list.sort()
        bg_img = cv2.imread(bg_base + i + "/" + i + bg_names[bg_arg])
        bg_HSV = cv2.cvtColor(bg_img, cv2.COLOR_BGR2HSV)
        #cv2.imwrite(result_path + os.path.splitext(i + bg_names[bg_arg])[0] + ".png", bg_HSV)
        for j in file_list:
            img = cv2.imread(j)
            img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            #cv2.imwrite(result_path + os.path.splitext(os.path.basename(j))[0] + ".png", img_HSV)
            img_G_threshold = cv2.inRange(img_HSV, (th_dic[i]["low_H"][bg_arg], th_dic[i]["low_S"][bg_arg], th_dic[i]["low_V"][bg_arg]), (th_dic[i]["high_H"][bg_arg], th_dic[i]["high_S"][bg_arg], th_dic[i]["high_V"][bg_arg]))
            img_H_dif = ((np.absolute(img_HSV[:,:,0].astype("int")-bg_HSV[:,:,0].astype("int"))) < threshold)*255
            #img_H_dif[700:2500,1000:4300] = img_G_threshold[700:2500,1000:4300]
            img_H_dif[700:3300,1000:4300] = img_G_threshold[700:3300,1000:4300]
            img_binary = (255 - img_H_dif).astype("uint8")       
            #cv2.imwrite(result_path + os.path.splitext(os.path.basename(j))[0] + "_crop.png", img_binary)
            contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            idx_max = max(enumerate(contours), key=lambda contour:cv2.contourArea(contour[1]))[0]
            x, y, width, height = cv2.boundingRect(contours[idx_max])
            x_min_list.append(x)
            y_min_list.append(y)
            x_max_list.append(x+width)
            y_max_list.append(y+height)
            contours2 = list(filter(lambda x: cv2.contourArea(x) >= 10000, contours))
            contour_img = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)
            #cv2.drawContours(img_mask, contours, idx_max, (255, 255, 255), -1)
            #contour_img = np.zeros_like(img)
            #cv2.drawContours(contour_img, contours, idx_max, (255, 255, 255), -1)
            cv2.drawContours(contour_img, contours2, -1, (255, 255, 255), -1)
            contour_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
            #img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            #cv2.imwrite(result_path + os.path.splitext(os.path.basename(j))[0] + "_crop.png", img_binary)
            cv2.imwrite(full_folder_c + os.path.splitext(os.path.basename(j))[0] + "_mask.png", contour_img)

            """
            #kmean        
            img_2d = np.float32(img.reshape((-1,3)))
            ret,label,center=cv2.kmeans(img_2d,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            result_image = res.reshape((img.shape))
            cv2.imwrite(result_path + os.path.splitext(os.path.basename(j))[0] + "_crop_km.png", result_image)
            """


    x_min = int(min(x_min_list))
    y_min = int(min(y_min_list))
    x_max = int(max(x_max_list))
    y_max = int(max(y_max_list))

    #add margin
    margin = 100
    x_min = x_min - margin
    y_min = y_min - margin
    x_max = x_max + margin
    y_max = y_max + margin

    if x_min > mtx_x_min:
        x_min = mtx_x_min
        print("xmin")    
    if x_max < mtx_x_max:
        x_max = mtx_x_max
        print("xmax")
    if y_min > mtx_y_min:
        y_min = mtx_y_min    
        print("ymin")
    if y_max < mtx_y_max:
        y_max = mtx_y_max
        print("ymax")
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > 5472:
        x_max = 5472
    if y_max > 3648:
        y_max = 3648
    
    offset.append([x_min,y_min])
    crop_img_size.append([x_max - x_min,y_max - y_min])

    if make_crop_images:
        dst_folder = base_folder + "/crop/" + str(species_number).zfill(3) + "-" + name
        img_folder = dst_folder + "/images/" 
        cor_folder = dst_folder + "/scan_images/"
        mask_folder = dst_folder + "/mask/"
        cor_folder_0 = cor_folder + "/0/"
        cor_folder_20 = cor_folder + "/20/"
        try:
            os.makedirs(dst_folder)
        except FileExistsError:
            pass
        try:
            os.makedirs(img_folder)
        except FileExistsError:
            pass
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
        try:
            os.makedirs(mask_folder)
        except FileExistsError:
            pass

        for count, i in enumerate(sd_labels):

            img_folder_c = img_folder + i +'/'
            try:
                os.makedirs(img_folder_c)
            except FileExistsError:
                pass

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

            mask_folder_c = mask_folder + i +'/'
            try:
                os.makedirs(mask_folder_c)
            except FileExistsError:
                pass

            f = original_folder_list[species_number - 1] + "/images/" + i + "/*.JPG"
            file_list = glob.glob(f)
            for j in file_list:
                img = cv2.imread(j)
                cv2.imwrite(img_folder_c + os.path.splitext(os.path.basename(j))[0] + ".png", img[y_min:y_max,x_min:x_max])

            f = original_folder_list[species_number - 1] + "/scan_images/0/" + i + "/*.JPG"
            file_list = glob.glob(f)
            for j in file_list:
                img = cv2.imread(j)
                cv2.imwrite(cor_folder_0_c + os.path.splitext(os.path.basename(j))[0] + ".png", img[y_min:y_max,x_min:x_max])

            f = original_folder_list[species_number - 1] + "/scan_images/20/" + i + "/*.JPG"
            file_list = glob.glob(f)
            for j in file_list:
                img = cv2.imread(j)
                cv2.imwrite(cor_folder_20_c + os.path.splitext(os.path.basename(j))[0] + ".png", img[y_min:y_max,x_min:x_max])

            f = full_dst_folder + '/' + i + "/*.png"
            file_list = glob.glob(f)
            for j in file_list:
                img = cv2.imread(j)
                cv2.imwrite(mask_folder_c + os.path.splitext(os.path.basename(j))[0] + ".png", img[y_min:y_max,x_min:x_max])

    print(str(int((time.time()- time_sta)//60)) + " min " + str(int((time.time()- time_sta)%60)) + " sec")
"""
out_file = '/multiview/datasets/360camera/butterfly/crop_pram.h5'
f_h5py = h5py.File(out_file, 'w')
f_h5py.create_group("crop")
f_h5py["crop"].create_dataset('offset', data=offset)
f_h5py["crop"].create_dataset('img_size', data=crop_img_size)
f_h5py.close()
"""
out_file = '/data2/suguru/datasets/360camera/butterfly/crop_pram.h5'
f_h5py = h5py.File(out_file, 'r')
ofs = f_h5py["crop/offset"][:]
ims = f_h5py["crop/img_size"][:]

ofs[14] = offset[0]
ims[14] = crop_img_size[0]
ofs[71] = offset[1]
ims[71] = crop_img_size[1]
f_h5py.close()

out_file = '/data2/suguru/datasets/360camera/butterfly/crop_pram_new.h5'
f_h5py = h5py.File(out_file, 'w')
f_h5py.create_group("crop")
f_h5py["crop"].create_dataset('offset', data=ofs)
f_h5py["crop"].create_dataset('img_size', data=ims)
f_h5py.close()