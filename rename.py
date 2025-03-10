import shutil
import glob
import os
sd_labels = ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']
base_folder = "/data2/suguru/datasets/360camera/butterfly/original/"


currentfolder ="110-Nymphalis_l-album-001"

targetfolder ="110-Nymphalis_l_album-001"

#os.rename(base_folder + currentfolder, base_folder + targetfolder)

replace_text = "Nymphalis_l-album"
dest_text = "Nymphalis_l_album"
#breakpoint()

for i in sd_labels:
    f = base_folder + targetfolder + "/images/" + i + "/*.JPG"
    f_list = glob.glob(f)
    for j in f_list:
        x = j.replace(replace_text, dest_text)
        os.rename(j, x)

for i in sd_labels:
    f = base_folder + targetfolder + "/scan_images/0/" + i + "/*.JPG"
    f_list = glob.glob(f)
    for j in f_list:
        x = j.replace(replace_text, dest_text)
        os.rename(j, x)

for i in sd_labels:
    f = base_folder + targetfolder + "/scan_images/20/" + i + "/*.JPG"
    f_list = glob.glob(f)
    for j in f_list:
        x = j.replace(replace_text, dest_text)
        os.rename(j, x)


base_folder = "/data2/suguru/datasets/360camera/butterfly/crop/"
#os.rename(base_folder + currentfolder, base_folder + targetfolder)
for i in sd_labels:
    f = base_folder + targetfolder + "/images/" + i + "/*.png"
    f_list = glob.glob(f)
    for j in f_list:
        x = j.replace(replace_text, dest_text)
        os.rename(j, x)

for i in sd_labels:
    f = base_folder + targetfolder + "/mask/" + i + "/*.png"
    f_list = glob.glob(f)
    for j in f_list:
        x = j.replace(replace_text, dest_text)
        os.rename(j, x)

for i in sd_labels:
    f = base_folder + targetfolder + "/scan_images/0/" + i + "/*.png"
    f_list = glob.glob(f)
    for j in f_list:
        x = j.replace(replace_text, dest_text)
        os.rename(j, x)

for i in sd_labels:
    f = base_folder + targetfolder + "/scan_images/20/" + i + "/*.png"
    f_list = glob.glob(f)
    for j in f_list:
        x = j.replace(replace_text, dest_text)
        os.rename(j, x)


f = "/data2/suguru/datasets/360camera/butterfly/undistort/" + targetfolder + "/*/*/*"
base_folder = "/data2/suguru/datasets/360camera/butterfly/undistort/" 
os.rename(base_folder + currentfolder, base_folder + targetfolder)
f_list = glob.glob(f)
for j in f_list:
    x = j.replace(replace_text, dest_text)
    os.rename(j, x)

f = "/data2/suguru/datasets/360camera/butterfly/crop_undistort/" + targetfolder + "/*/*/*"
base_folder = "/data2/suguru/datasets/360camera/butterfly/crop_undistort/" 
os.rename(base_folder + currentfolder, base_folder + targetfolder)

f_list = glob.glob(f)
for j in f_list:
    x = j.replace(replace_text, dest_text)
    os.rename(j, x)

f = "/data2/suguru/datasets/360camera/butterfly/crop_mask_undistort/" + targetfolder + "/*/*"
base_folder = "/data2/suguru/datasets/360camera/butterfly/crop_mask_undistort/" 
os.rename(base_folder + currentfolder, base_folder + targetfolder)

f_list = glob.glob(f)
for j in f_list:
    x = j.replace(replace_text, dest_text)
    os.rename(j, x)


f = "/data2/suguru/datasets/360camera/butterfly/full_mask/" + targetfolder + "/*/*"
base_folder = "/data2/suguru/datasets/360camera/butterfly/full_mask/" 
#os.rename(base_folder + currentfolder, base_folder + targetfolder)

f_list = glob.glob(f)
for j in f_list:
    x = j.replace(replace_text, dest_text)
    os.rename(j, x)

f = "/data2/suguru/datasets/360camera/butterfly/full_mask_undistort/" + targetfolder + "/*/*"
base_folder = "/data2/suguru/datasets/360camera/butterfly/full_mask_undistort/" 
os.rename(base_folder + currentfolder, base_folder + targetfolder)

f_list = glob.glob(f)
for j in f_list:
    x = j.replace(replace_text, dest_text)
    os.rename(j, x)

f = "/data2/suguru/datasets/360camera/butterfly/correspondence/" + targetfolder + "/*.pcd"
base_folder = "/data2/suguru/datasets/360camera/butterfly/correspondence/" 
os.rename(base_folder + currentfolder, base_folder + targetfolder)

f_list = glob.glob(f)
for j in f_list:
    x = j.replace(replace_text, dest_text)
    os.rename(j, x)


"""
base_folder2 = "/multiview/datasets/360camera/bg_images/"
dest_text = ""
for i in sd_labels:
    f = base_folder2 + i + "/*.JPG"
    f_list = glob.glob(f)
    f_list.sort()
    os.rename(f_list[0], base_folder2 + i + "/" + i + "_bg_f20e-1.JPG")
    os.rename(f_list[1], base_folder2 + i + "/" + i + "_bg_f32e-1.JPG")
    os.rename(f_list[2], base_folder2 + i + "/" + i + "_bg_f25e-1.JPG")


"""