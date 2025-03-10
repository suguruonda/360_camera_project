import numpy as np
import cv2 as cv
import glob
import os
import h5py
from CB_calibration import CB
import bundle_adjustment_8poses

def cameraGL(num):
    if num in [1,3,6,]:
        return 0
    elif num in [2,4,5]:
        return 1
        
def store_param(st_dict, camera, R, T, CP, P3D, MTX):
    c_indx = int(camera[-1]) - 1
    if st_dict["bool"][c_indx] == False:
        st_dict["bool"][c_indx] = True
        st_dict["MTX"][c_indx] = MTX
        str_num = "1"
    else:
        str_num = "2"

    st_dict["R" + str_num][c_indx] = R
    st_dict["T" + str_num][c_indx] = T
    st_dict["CP" + str_num][c_indx] = CP
    st_dict["P3D" + str_num][c_indx] = P3D

    return st_dict


if __name__ == "__main__":
    
    cparam_file = "camera_pram_2.h5"
    f = h5py.File(cparam_file, "r")   
    
    pose_list = ["camera1and2and3and4", "camera3and4and5and6","camera5and6and7and8",'camera8']
    pose_list.reverse()
    bool_dic = {'camera1':False,'camera2':False,'camera3':False,'camera4':False,'camera5':False,'camera6':False,'camera7':False,'camera8':False}
    rvec_dic = {'camera1':0,'camera2':0,'camera3':0,'camera4':0,'camera5':0,'camera6':0,'camera7':0,'camera8':0}
    tvec_dic = {'camera1':0,'camera2':0,'camera3':0,'camera4':0,'camera5':0,'camera6':0,'camera7':0,'camera8':0}
    
    img_path = 'data/8cameras/'

    stores_dict = {"bool" : [False for i in range(8)],\
                    "R1" : [None for i in range(8)],\
                    "R2" : [None for i in range(8)],\
                    "T1" : [None for i in range(8)],\
                    "T2" : [None for i in range(8)],\
                    "CP1" : [None for i in range(8)],\
                    "CP2" : [None for i in range(8)],\
                    "P3D1" : [None for i in range(8)],\
                    "P3D2" : [None for i in range(8)],\
                    "MTX" : [None for i in range(8)]}

    for i in pose_list:
        folders = glob.glob(img_path + i + '/*')
        folders.sort(reverse=True)
        r_dif = [None,None]
        t_dif = [None,None]
        for num, j in enumerate(folders):
            camera = os.path.basename(j)
            c_num = int(camera[-1])
            if i == 'camera8':
                c = CB(j,4,3,15)
            else:
                c = CB(j,6,4,9.95)
            c.mtx = np.array(f[camera]['mtx'], dtype=np.float64)
            c.dist = np.array(f[camera]['dist'], dtype=np.float64)
            if not(c.find_CB_coners(4)):
                c.find_CB_coners(8)
                print(8)
            if i == "camera1and2and3and4" and (c_num == 3 or c_num == 1):
                c.flip_corners()
            c.cal_extrinsic_parameters()
            c.undistortPoint()
            c.eval_distortion()
            
            if i == 'camera8':
                rvec_dic[camera], _ = cv.Rodrigues(np.array(c.rvecs[0], dtype=np.float64))
                tvec_dic[camera] = np.array(c.tvecs[0], dtype=np.float64)
                bool_dic[camera] = True
                stores_dict = store_param(stores_dict,\
                                camera,\
                                np.array(c.rvecs[0],dtype=np.float64),\
                                np.array(c.tvecs[0], dtype=np.float64),\
                                c.imgpoints_undistorted[0],\
                                c.objpoints2[0],\
                                c.mtx)
            else:
                if bool_dic[camera] == True:
                    r1 = rvec_dic[camera]
                    t1 = tvec_dic[camera]
                    r2, _ = cv.Rodrigues(np.array(c.rvecs[0], dtype=np.float64))
                    t2 = np.array(c.tvecs[0], dtype=np.float64)

                    stores_dict = store_param(stores_dict,\
                                    camera,\
                                    np.array(c.rvecs[0],dtype=np.float64),\
                                    np.array(c.tvecs[0], dtype=np.float64),\
                                    c.imgpoints_undistorted[0],\
                                    c.objpoints2[0],\
                                    c.mtx)
                    
                    if i == "camera5and6and7and8":
                        r_dif = np.dot(r2.T, r1)
                        t_dif = np.dot(r2.T, t1 - t2)
                    else:
                        r_dif[cameraGL(c_num)] = np.dot(r2.T, r1)
                        t_dif[cameraGL(c_num)] = np.dot(r2.T, t1 - t2)
                else:
                    r3, _ = cv.Rodrigues(np.array(c.rvecs[0], dtype=np.float64))
                    t3 = np.array(c.tvecs[0], dtype=np.float64)

                    stores_dict = store_param(stores_dict,\
                                    camera,\
                                    np.array(c.rvecs[0],dtype=np.float64),\
                                    np.array(c.tvecs[0], dtype=np.float64),\
                                    c.imgpoints_undistorted[0],\
                                    c.objpoints2[0],\
                                    c.mtx)

                    if i == "camera5and6and7and8":
                        r_d = r_dif
                        t_d = t_dif
                    else:
                        r_d = r_dif[cameraGL(c_num)]
                        t_d = t_dif[cameraGL(c_num)]
                    rvec_dic[camera] =  np.dot(r3, r_d)
                    tvec_dic[camera] = t3 + np.dot(r3, t_d) 
                    bool_dic[camera] = True


    #store dummy data we don't use
    stores_dict = store_param(stores_dict,\
                    "camera1",\
                    np.array(c.rvecs[0],dtype=np.float64),\
                    np.array(c.tvecs[0], dtype=np.float64),\
                    c.imgpoints_undistorted[0],\
                    c.objpoints2[0],\
                    c.mtx)    
    stores_dict = store_param(stores_dict,\
                    "camera2",\
                    np.array(c.rvecs[0],dtype=np.float64),\
                    np.array(c.tvecs[0], dtype=np.float64),\
                    c.imgpoints_undistorted[0],\
                    c.objpoints2[0],\
                    c.mtx)    
    stores_dict = store_param(stores_dict,\
                    "camera7",\
                    np.array(c.rvecs[0],dtype=np.float64),\
                    np.array(c.tvecs[0], dtype=np.float64),\
                    c.imgpoints_undistorted[0],\
                    c.objpoints2[0],\
                    c.mtx)    





    Rs, R_s, Ts, T_s = bundle_adjustment_8poses.optimize(stores_dict)

    rvec_dic_opt = {'camera1':0,'camera2':0,'camera3':0,'camera4':0,'camera5':0,'camera6':0,'camera7':0,'camera8':0}
    tvec_dic_opt = {'camera1':0,'camera2':0,'camera3':0,'camera4':0,'camera5':0,'camera6':0,'camera7':0,'camera8':0}

    for num, i in enumerate(list(bool_dic.keys())):
        rvec_dic_opt[i] = Rs[num]
        tvec_dic_opt[i] = Ts[num].reshape(3,1)


    rotate_rvec_dic = {'camera1':[],'camera2':[],'camera3':[],'camera4':[],'camera5':[],'camera6':[],'camera7':[],'camera8':[]}
    rotate_tvec_dic = {'camera1':[],'camera2':[],'camera3':[],'camera4':[],'camera5':[],'camera6':[],'camera7':[],'camera8':[]}
    rotate_rvec_dic_opt = {'camera1':[],'camera2':[],'camera3':[],'camera4':[],'camera5':[],'camera6':[],'camera7':[],'camera8':[]}
    rotate_tvec_dic_opt = {'camera1':[],'camera2':[],'camera3':[],'camera4':[],'camera5':[],'camera6':[],'camera7':[],'camera8':[]}
    w_rvec_dic = {'camera1':[],'camera2':[],'camera3':[],'camera4':[],'camera5':[],'camera6':[],'camera7':[],'camera8':[]}
    w_tvec_dic = {'camera1':[],'camera2':[],'camera3':[],'camera4':[],'camera5':[],'camera6':[],'camera7':[],'camera8':[]}
    w_rvec_dic_opt = {'camera1':[],'camera2':[],'camera3':[],'camera4':[],'camera5':[],'camera6':[],'camera7':[],'camera8':[]}
    w_tvec_dic_opt = {'camera1':[],'camera2':[],'camera3':[],'camera4':[],'camera5':[],'camera6':[],'camera7':[],'camera8':[]}
    
    for i in list(bool_dic.keys()):   
        rotate_rvec_dic[i].append(rvec_dic[i])
        rotate_tvec_dic[i].append(tvec_dic[i]) 
        w_rvec_dic[i].append(rvec_dic[i].T)
        w_tvec_dic[i].append(np.dot(-rvec_dic[i].T, tvec_dic[i]))

        rotate_rvec_dic_opt[i].append(rvec_dic_opt[i])
        rotate_tvec_dic_opt[i].append(tvec_dic_opt[i]) 
        w_rvec_dic_opt[i].append(rvec_dic_opt[i].T)
        w_tvec_dic_opt[i].append(np.dot(-rvec_dic_opt[i].T, tvec_dic_opt[i]))
    

    # camera pose visualization

    from camera_visualization import camera_visualizer

    visualizer = camera_visualizer()
    #poses = np.zeros((len(rvec_dic[camera]), 4, 4))
    poses = np.zeros((8, 4, 4))

    color_dic = {'camera1':'red','camera2':'orange','camera3':'yellow','camera4':'lawngreen','camera5':'cyan','camera6':'blue','camera7':'blueviolet','camera8':'magenta'}

    for num,i in enumerate(list(bool_dic.keys())):
        mat = np.concatenate([w_rvec_dic[i][0], w_tvec_dic[i][0]],  axis=1)
        tmp = np.array([0,0,0,1])
        mat4 = np.vstack((mat, tmp.T))
        poses[num] = mat4
    visualizer.plot_camera_scene(poses, 30, color_dic["camera6"], "data", False)

    for num,i in enumerate(list(bool_dic.keys())):
        mat = np.concatenate([w_rvec_dic_opt[i][0], w_tvec_dic_opt[i][0]],  axis=1)
        tmp = np.array([0,0,0,1])
        mat4 = np.vstack((mat, tmp.T))
        poses[num] = mat4
    visualizer.plot_camera_scene(poses, 30, color_dic["camera1"], "Optimized", False)

    visualizer.show()

    out_file = "camera_pram_no_opt.h5"
    f_out = h5py.File(out_file, "w")   
    for i in range(1,9):
        grp = 'camera'+str(i)
        f_out.create_group(grp)
        f_out[grp].create_dataset('mtx', data=np.array(f[grp]['mtx'], dtype=np.float64))
        f_out[grp].create_dataset('dist', data=np.array(f[grp]['dist'], dtype=np.float64))
        f_out[grp].create_dataset('rvec', data=rotate_rvec_dic[grp])
        f_out[grp].create_dataset('tvec', data=rotate_tvec_dic[grp])
    

    f_out.close()

    out_file = "camera_pram_optimized.h5"
    f_out = h5py.File(out_file, "w")   
    for i in range(1,9):
        grp = 'camera'+str(i)
        f_out.create_group(grp)
        f_out[grp].create_dataset('mtx', data=np.array(f[grp]['mtx'], dtype=np.float64))
        f_out[grp].create_dataset('dist', data=np.array(f[grp]['dist'], dtype=np.float64))
        f_out[grp].create_dataset('rvec', data=rotate_rvec_dic_opt[grp])
        f_out[grp].create_dataset('tvec', data=rotate_tvec_dic_opt[grp])
    

    f_out.close()
    f.close()