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

def DLT(coordinates, p_s):
    A = []
    for i in range(len(coordinates)):
        x = coordinates[i][0]
        y = coordinates[i][1]
        p1 = p_s[i][0,:]
        p2 = p_s[i][1,:]
        p3 = p_s[i][2,:]
        A.append(y*p3 - p2)
        A.append(p1 - x*p3)

    A = np.array(A).reshape((-1,4))
    B = A.transpose() @ A
    U, s, Vh = np.linalg.svd(B, full_matrices = False)
 
    return Vh[-1,:]

base_path = "/data2/suguru/datasets/360camera/butterfly/"
sd_labels = ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']

original_folder_list = glob.glob(base_path + "original/*")
original_folder_list.sort()

#crop_file = '/data2/suguru/datasets/360camera/butterfly/crop_pram.h5'
crop_file = '/data2/suguru/datasets/360camera/butterfly/crop_pram_undistort_new.h5'
f_crop = h5py.File(crop_file, 'r')

#cam_para_file = '/data2/suguru/datasets/360camera/camera_pram.h5'
#cam_para_file = '/data2/suguru/datasets/360camera/camera_pram_optimized.h5'
#cam_para_file = '/data2/suguru/datasets/360camera/camera_pram_2_no180.h5'
#cam_para_file = '/data2/suguru/datasets/360camera/camera_pram_2.h5'
cam_para_file = '/data2/suguru/datasets/360camera/camera_pram_2_no180_2_opt.h5'
#cam_para_file = '/data2/suguru/datasets/360camera/camera_pram_2_no180_2_opt_test_vvs.h5'
#cam_para_file = '/data2/suguru/datasets/360camera/camera_pram_2_no180_2_opt_test_vlm.h5'
f_param = h5py.File(cam_para_file, 'r')

for i in range(1):
    species_number = i + 1

    #21
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
        cor_0_dic[c][:,0] += x
        cor_20_dic[c][:,0] += x
        cor_0_dic[c][:,1] += y
        cor_20_dic[c][:,1] += y 


        rvec = f_param[c]['rvec'][0]
        tvec = f_param[c]['tvec'][0]
        #p_0_dic[c] = mtx @ np.concatenate([rvec,tvec], axis=1)
        p_list[count] = mtx @ np.concatenate([rvec,tvec], axis=1)
        #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (5472,3648), 1, (5472,3648))
        #temp = cv2.undistortPoints(cor_0_dic[c][:,0:2].T, mtx, dist, None, newcameramtx)
        temp = cv2.undistortImagePoints(cor_0_dic[c][:,0:2].T, mtx, dist)
        #undist_0_dic[c] = np.concatenate((temp.flatten().reshape(-1,2),cor_0_dic[c][:,2:]),axis=1)
        undist_0_dic[c] = temp.flatten().reshape(-1,2)

        rvec = f_param[c]['rvec'][20]
        tvec = f_param[c]['tvec'][20]
        #p_20_dic[c] = mtx @  np.concatenate([rvec,tvec], axis=1)
        p_list[count + 8] = mtx @ np.concatenate([rvec,tvec], axis=1)
        
        #temp = cv2.undistortPoints(cor_20_dic[c][:,0:2].T, mtx, dist, None, newcameramtx)
        temp = cv2.undistortImagePoints(cor_20_dic[c][:,0:2].T, mtx, dist)
        #undist_20_dic[c] = np.concatenate((temp.flatten().reshape(-1,2),cor_20_dic[c][:,2:]),axis=1)
        undist_20_dic[c] = temp.flatten().reshape(-1,2)

        num_0.append(len(cor_0_dic[c][:,2]))
        num_20.append(len(cor_20_dic[c][:,2]))


    map_bottom_0 = np.zeros((max(num_0[0:4]),4), dtype=int)
    map_top_0 = np.zeros((max(num_0[4:]),4), dtype=int)
    map_bottom_20 = np.zeros((max(num_20[0:4]),4), dtype=int)
    map_top_20 = np.zeros((max(num_20[4:]),4), dtype=int)

    for count, c in enumerate(sd_labels):
        if count < 4:
            map_bottom_0[0:num_0[count],count] = cor_0_dic[c][:,2]
            map_bottom_20[0:num_20[count],count] = cor_20_dic[c][:,2]
        else:
            count2 = count - 4
            map_top_0[0:num_0[count],count2] = cor_0_dic[c][:,2]
            map_top_20[0:num_20[count],count2] = cor_20_dic[c][:,2]

    #matching
    coordinates_3d_list = []
    coordinates_3d_list_noba = []
    rgb_list = []
    b_images_0 = []
    b_images_20 = []
    for count, c in enumerate(sd_labels):
        img_name = original_folder_list[species_index] + "/images/" + "/" + c + "/" + name[4:] + "-" + c + "-" + "00" + ".JPG"
        b_images_0.append(cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB))
        img_name = original_folder_list[species_index] + "/images/" + "/" + c + "/" + name[4:] + "-" + c + "-" + "20" + ".JPG"
        b_images_20.append(cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB))

    temp_map_list = [map_bottom_0,map_top_0,map_bottom_20,map_top_20]
    camera_id_offset = [0,4,0,4]

    zp = np.array([0.0,0.0])
    coordinates_2d_list = []
    distorted_coordinates_2d_list = []
    mask_list = []

    #for k in [0,1]:
    for k in range(4):
        temp_map = temp_map_list[k]
        vals = np.trim_zeros(np.unique(temp_map))
        match_list = []
        for count, i in enumerate(vals):
            match = np.argwhere(temp_map==i)
            if match.shape[0] > 1:
                match_list.append(match)

        for i in match_list:
            coordinates_array = np.array([zp for i in range(16)])
            distorted_coordinates_array = np.array([zp for i in range(16)])
            mask = np.array([0.0 for i in range(16)])
            coordinates = []
            p_s = []
            rgb_s = []
            #if len(i) == 4:
            for j in range(len(i)):
                point_id = i[j][0]
                camera_id = i[j][1] + camera_id_offset[k]
                if k < 2:
                    coordinates_array[camera_id] = undist_0_dic[sd_labels[camera_id]][point_id]
                    distorted_coordinates_array[camera_id] = cor_0_dic[sd_labels[camera_id]][point_id][0:2]
                    mask[camera_id] = 1.0
                    coordinates.append(undist_0_dic[sd_labels[camera_id]][point_id])
                    #x_pos = int(cor_0_dic[sd_labels[camera_id]][point_id][0] + 0.5)
                    #y_pos = int(cor_0_dic[sd_labels[camera_id]][point_id][1] + 0.5)
                    x_pos = int(cor_0_dic[sd_labels[camera_id]][point_id][0])
                    y_pos = int(cor_0_dic[sd_labels[camera_id]][point_id][1])
                    rgb_s.append(b_images_0[camera_id][y_pos,x_pos])
                    p_s.append(p_list[camera_id])
                else:
                    coordinates_array[camera_id + 8] = undist_20_dic[sd_labels[camera_id]][point_id]
                    distorted_coordinates_array[camera_id + 8] = cor_20_dic[sd_labels[camera_id]][point_id][0:2]
                    mask[camera_id + 8] = 1.0
                    coordinates.append(undist_20_dic[sd_labels[camera_id]][point_id])
                    #x_pos = int(cor_20_dic[sd_labels[camera_id]][point_id][0] + 0.5)
                    #y_pos = int(cor_20_dic[sd_labels[camera_id]][point_id][1] + 0.5)
                    x_pos = int(cor_20_dic[sd_labels[camera_id]][point_id][0])
                    y_pos = int(cor_20_dic[sd_labels[camera_id]][point_id][1])
                    rgb_s.append(b_images_20[camera_id][y_pos,x_pos])
                    p_s.append(p_list[camera_id + 8])

            X = DLT(coordinates,p_s)
            #X = cv2.triangulatePoints(p_s[0], p_s[1], coordinates[0].reshape(2,1), coordinates[1].reshape(2,1))
            #X /= X[3]
            rgb = np.average(np.array(rgb_s), axis=0)
            coordinates_3d_list_noba.append(X[0:3]/X[-1])
            coordinates_3d_list.append(X)
            coordinates_2d_list.append(coordinates_array)
            distorted_coordinates_2d_list.append(distorted_coordinates_array)
            mask_list.append(mask)
            rgb_list.append(rgb)

    #coordinates_3d_list = np.array(coordinates_3d_list).flatten().reshape(-1,3)
    coordinates_3d_list = np.array(coordinates_3d_list)
    p_list = np.array(p_list)
    coordinates_2d_list = np.array(coordinates_2d_list)
    distorted_coordinates_2d_list = np.array(distorted_coordinates_2d_list)
    mask_list = np.array(mask_list)
    rgb_list = np.array(rgb_list)

    """
    pair_0 = 0
    pair_1 = 1
    c_0 = sd_labels[pair_0 % 8]
    c_1 = sd_labels[pair_1 % 8]
    c_point_0 = distorted_coordinates_2d_list[(mask_list[:,pair_0]+mask_list[:,pair_1])==2,pair_0,:]
    c_point_1 = distorted_coordinates_2d_list[(mask_list[:,pair_0]+mask_list[:,pair_1])==2,pair_1,:]

    retval, E, Rdif, tdif, mask_ = cv2.recoverPose(c_point_0, c_point_1, f_param[c_0]['mtx'][:], f_param[c_0]['dist'][:], f_param[c_1]['mtx'][:], f_param[c_0]['dist'][:] )

    def add_relativeRT(r,t,rdif,tdif):
        R = np.dot(r, rdif)
        T = t + np.dot(r, tdif)
        return R,T

    def add_relativeRT2(r,t,rdif,tdif):
        R = np.dot(rdif,r)
        T = tdif + np.dot(rdif, t)
        return R,T

    mtx_0 = f_param[c_0]['mtx'][:]
    rvec_0 = f_param[c_0]['rvec'][0]
    tvec_0 = f_param[c_0]['tvec'][0]

    mtx_1 = f_param[c_1]['mtx'][:]
    rvec_1 = f_param[c_1]['rvec'][0]
    tvec_1 = f_param[c_1]['tvec'][0]

    print(Rdif, tdif)

    n_rvec_1, n_tvec_1 = add_relativeRT(rvec_0,tvec_0,Rdif,tdif)
    print(n_rvec_1, n_tvec_1)
    n_rvec_1, n_tvec_1 = add_relativeRT2(rvec_0,tvec_0,Rdif,tdif)
    print(n_rvec_1, n_tvec_1)
    print(rvec_1, tvec_1)
    """


    mask_list = mask_list.T
    c_input = coordinates_2d_list
    coordinates_3d_list = coordinates_3d_list.T
    ba_bool = False

    if ba_bool:
        coordinates_3d_list_ba = bundle_adjustment.main(c_input,mask_list,p_list,coordinates_3d_list)




    #imgpoints2, _ = cv.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)


    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    if ba_bool:
        pcd.points = o3d.utility.Vector3dVector(coordinates_3d_list_ba)
    else:
        pcd.points = o3d.utility.Vector3dVector(coordinates_3d_list_noba)

    pcd.colors = o3d.utility.Vector3dVector(rgb_list/255)

    import math
    import itertools
    bounds = [[-math.inf, math.inf], [-math.inf, math.inf], [-20, 20]]
    bounds = [[-math.inf, math.inf], [-math.inf, math.inf], [-60, 60]]
    bounding_box_points = list(itertools.product(*bounds))
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points))

    # Crop the point cloud using the bounding box:
    pcd_croped = pcd.crop(bounding_box)

    o3d.visualization.draw_geometries([pcd_croped])

    #pcd.normals = o3d.utility.Vector3dVector(normals)
    #koko
    #cl, ind = pcd_croped.remove_statistical_outlier(nb_neighbors=40, std_ratio=4)

    #cl, ind = pcd.remove_radius_outlier(np_points=20, radius=0.5)
    """
    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])

    #display_inlier_outlier(pcd,ind)
    #o3d.visualization.draw_geometries([pcd])
    #o3d.visualization.draw_geometries([pcd_croped.select_by_index(ind)])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    #vis.create_window('test', 5472, 3648, 0, 0)
    #koko

    vis.add_geometry(pcd_croped)
    #vis.add_geometry(pcd)
    #vis.add_geometry(pcd_croped.select_by_index(ind))
    view_control = vis.get_view_control()
    pinhole_parameters = view_control.convert_to_pinhole_camera_parameters()
    extrinsic = pinhole_parameters.extrinsic.copy()
    #intrinsic = pinhole_parameters.intrinsic.intrinsic_matrix.copy()

    c = "camera7"
    rvec = f_param[c]['rvec'][20]
    tvec = f_param[c]['tvec'][20]
    print(extrinsic)
    #extrinsic[0:3,:] = np.concatenate([rvec.T, np.dot(-rvec.T, tvec)], axis=1)
    extrinsic[0:3,:] = np.concatenate([rvec, tvec], axis=1)
    #extrinsic[:3,3] = np.array([-500,500,2000])
    print(extrinsic)
    #intrinsic[:,:] = f_param[c]['mtx'][:]

    pinhole_parameters.extrinsic = extrinsic
    #pinhole_parameters.intrinsic.intrinsic_matrix = intrinsic
    vis.get_view_control().convert_from_pinhole_camera_parameters(pinhole_parameters)
    #view_control.convert_from_pinhole_camera_parameters(pinhole_parameters)
    vis.run()
    """

    o3d.io.write_point_cloud(cor_folder + "/" + name + ".pcd", pcd)
