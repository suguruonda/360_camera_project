import glob
import os
import glob
import shutil
import time



sd_labels = ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']
base_folder = "/data2/suguru/datasets/360camera/butterfly/"
#base_folder = "/home/ondas/nobackup/archive/360camera/butterfly/"
original_folder_list = glob.glob(base_folder + "crop_undistort/*")
original_folder_list.sort()
out_base = base_folder + "colmap/crop_undistort/"

for k, ele in enumerate(original_folder_list[0:2]):
    name = os.path.basename(ele)
    dst_folder_top = out_base + name + "/top/images/"
    try:
        os.makedirs(dst_folder_top)
    except FileExistsError:
        pass

    dst_folder_bottom = out_base + name + "/bottom/images/"
    try:
        os.makedirs(dst_folder_bottom)
    except FileExistsError:
        pass

    for i in sd_labels[0:4]:
        original_file_list = glob.glob(ele + "/images/" + i + "/*")
        for j in original_file_list:
            fname = os.path.basename(j)
            #os.symlink(j, dst_folder_top + fname)
            shutil.copyfile(j, dst_folder_top + fname)

    for i in sd_labels[4:]:
        original_file_list = glob.glob(ele + "/images/" + i + "/*")
        for j in original_file_list:
            fname = os.path.basename(j)
            #os.symlink(j, dst_folder_bottom + fname)
            shutil.copyfile(j, dst_folder_bottom + fname)
    