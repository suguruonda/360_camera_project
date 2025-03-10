import h5py
import cv2




cam_para_file = '/data2/suguru/datasets/360camera/camera_pram_final.h5'
f_param = h5py.File(cam_para_file, 'r+')
crop_file = '/data2/suguru/datasets/360camera/butterfly/crop_pram_undistort_new.h5'
f_crop = h5py.File(crop_file, 'r')

sd_labels = ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']
breakpoint()
for count, c in enumerate(sd_labels):
    mtx = f_param[c]['mtx'][:]
    dist = f_param[c]['dist'][:]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (5472,3648), 1, (5472,3648))
    f_param[c].create_dataset('mtx_undistort', data=newcameramtx)


f_param.create_group("crop")
f_param["crop"].create_dataset('offset', data=f_crop["crop/offset"][:])
f_param["crop"].create_dataset('img_size', data=f_crop["crop/img_size"][:])
f_param.close()
f_crop.close()