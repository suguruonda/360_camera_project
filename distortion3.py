import numpy as np
import cv2 as cv
import glob
import screeninfo
import os
import h5py

class CB_calbration:
    def __init__(self, img_path,c,r,cb_size):
        # termination criteria
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.c = c
        self.r = r
        #square size in mm
        self.cb_size = cb_size
        self.screen = screeninfo.get_monitors()[0]
        _objp = np.zeros((self.c*self.r,3), np.float32)
        _objp[:,:2] = np.mgrid[0:self.c,0:self.r].T.reshape(-1,2)
        self.objp = _objp * self.cb_size

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        if os.path.isdir(img_path):
            self.images = glob.glob(img_path + '/*.JPG')
            self.images.sort()
        elif os.path.isfile(img_path):
            self.images = [img_path]

        self.img_out_path = img_path + '_c/'
        self.img_size = 0
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = []
        self.tvecs = []

    def set_intrinsic_parameters(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist

    def find_CB_coners(self,scale, imshow_bool = False):
        count = 0
        for fname in self.images:
            img = cv.imread(fname)
            #img_ = cv.medianBlur(img,5)
            #img_ = cv.GaussianBlur(img,(5,5),0)
            #img_ = cv.bilateralFilter(img,9,75,75)
            img_resized = cv.resize(img, (img.shape[1]//scale,img.shape[0]//scale), interpolation=cv.INTER_AREA)
            #img_resized = cv.resize(img, (img.shape[1]//scale,img.shape[0]//scale), interpolation=cv.INTER_LANCZOS4)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            self.img_size = gray.shape
            gray_resized = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray_resized, (self.c,self.r), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                print(fname)
                count += 1
                self.objpoints.append(self.objp)
                corners2 = cv.cornerSubPix(gray,corners*scale, (11,11), (-1,-1), self.criteria)
                self.imgpoints.append(corners2)
                # Draw and display the corners
                if imshow_bool:
                    cv.drawChessboardCorners(img, (self.c,self.r), corners2, ret)
                    cv.namedWindow("img", cv.WINDOW_NORMAL)
                    cv.resizeWindow('img', self.screen.width, self.screen.height)
                    cv.imshow("img",img)
                    cv.waitKey(0)
        if imshow_bool:
            cv.destroyAllWindows()
        print(str(count) + " out of " + str(len(self.images)))
        return ret
        
    def drawAxes(self):
        count = 0
        for i,fname in enumerate(self.images):
            img = cv.imread(fname)
            cv.drawFrameAxes(img, self.mtx, self.dist, self.rvecs[i], self.tvecs[i],length=10, thickness=20)
            cv.namedWindow("img", cv.WINDOW_NORMAL)
            cv.resizeWindow('img', self.screen.width, self.screen.height)
            cv.imshow("img",img)
            cv.waitKey(0)

    def cal_camera_parameters(self):
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.img_size[::-1], None, None)

    def cal_extrinsic_parameters(self):
        self.ret, rvecs, tvecs = cv.solvePnP(np.array(self.objpoints, dtype=np.float64), np.reshape(np.array(self.imgpoints, dtype=np.float64), (1,self.c*self.r,2)), self.mtx, self.dist)
        self.rvecs.append(rvecs) 
        self.tvecs.append(tvecs)


    def undistortion(self,imshow_bool = False):
        for fname in self.images:
            img = cv.imread(fname)
            h,  w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
            dst = cv.undistort(img, self.mtx, self.dist, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            cv.imwrite(self.img_out_path + os.path.basename(fname), dst)
            if imshow_bool:
                cv.namedWindow("img", cv.WINDOW_NORMAL)
                cv.resizeWindow('img', self.screen.width, self.screen.height)
                cv.imshow("img",dst)
                cv.waitKey(0)
        if imshow_bool:
            cv.destroyAllWindows()
        
    def eval_distortion(self):
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print( "total error: {}".format(mean_error/len(self.objpoints)) )
        return mean_error/len(self.objpoints)

    def show_distortion_plot(self):
        pts_new, _ = cv.projectPoints(self.objpoints[0], self.rvecs[0], self.tvecs[0], self.mtx, self.dist)
        pts, _ = cv.projectPoints(self.objpoints[0], self.rvecs[0], self.tvecs[0], self.mtx, None)
        with open(self.img_out_path + 'distortion_sample.txt', 'w') as f:
            for i in range(pts.shape[0]):
                f.write("%f,%f,%f,%f" % (pts[i,0,0],pts[i,0,1],pts_new[i,0,0],pts_new[i,0,1]))
                f.write('\n')


if __name__ == "__main__":
    

    out_file = '/data2/suguru/datasets/360camera/camera_pram_2_no180_new.h5'
    #out_file = '/data2/suguru/datasets/360camera/camera_pram_2.h5'
    #out_file = '/data2/suguru/datasets/360camera/camera_pram_2_wc.h5'
    
    """
    f = h5py.File(out_file, 'w')
    
    img_path = '/multiview/datasets/360camera/cal/cb/dist/camera'
     
    for i in range(1,9):
        grp = 'camera'+str(i)
        f.create_group(grp)
        c = CB_calbration(img_path + str(i),14,10,17)
        c.find_CB_coners(4)
        c.cal_camera_parameters()
        f[grp].create_dataset('mtx', data=c.mtx)
        f[grp].create_dataset('dist', data=c.dist)
        c.undistortion()
        c.eval_distortion()
        #c.show_distortion_plot()
    
    
    #pose_list = ['camera1and2', 'camera1and3', 'camera2and4', 'camera3and6', 'camera4and5', 'camera5and7', 'camera6and8', 'camera7and8']
    pose_list = [ 'camera1and3','camera1and2', 'camera2and4', 'camera4and5', 'camera5and7', 'camera6and8', 'camera7and8', 'camera8']
    pose_list.reverse()
    """
    bool_dic = {'camera1':False,'camera2':False,'camera3':False,'camera4':False,'camera5':False,'camera6':False,'camera7':False,'camera8':False}
    """
    rvec_dic = {'camera1':0,'camera2':0,'camera3':0,'camera4':0,'camera5':0,'camera6':0,'camera7':0,'camera8':0}
    tvec_dic = {'camera1':0,'camera2':0,'camera3':0,'camera4':0,'camera5':0,'camera6':0,'camera7':0,'camera8':0}
    
    """
    f = h5py.File(out_file, 'r')
    """
    img_path = '/multiview/datasets/360camera/cal/cb/c_pose/'

    for i in pose_list:
        folders = glob.glob(img_path + i + '/*')
        if i != 'camera8':
            camera = os.path.basename(folders[0])
            if bool_dic[camera] == False:
                folders.reverse()
        for num, j in enumerate(folders):
            camera = os.path.basename(j)
            if i == 'camera8' or i == 'camera6and8' or i == 'camera7and8':
                c = CB_calbration(j,4,3,15)
            else:
                c = CB_calbration(j,6,5,17)
            c.mtx = np.array(f[camera]['mtx'], dtype=np.float64)
            c.dist = np.array(f[camera]['dist'], dtype=np.float64)
            if not(c.find_CB_coners(4)):
                c.find_CB_coners(8)
                print(8)
            c.cal_extrinsic_parameters()
            #c.drawAxes()
           
            c.eval_distortion()
            if i == 'camera8':
                rvec_dic[camera], _ = cv.Rodrigues(np.array(c.rvecs[0], dtype=np.float64))
                tvec_dic[camera] = np.array(c.tvecs[0], dtype=np.float64)
                x_180 = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
                y_180 = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
                #print(rvec_dic[camera])
                #print(tvec_dic[camera])
                rvec_dic[camera] = np.dot(rvec_dic[camera], x_180)
                bool_dic[camera] = True
            elif num == 0 and bool_dic[camera] == True:
                r1 = rvec_dic[camera]
                t1 = tvec_dic[camera]
                r2, _ = cv.Rodrigues(np.array(c.rvecs[0], dtype=np.float64))
                t2 = np.array(c.tvecs[0], dtype=np.float64)
                r_dif = np.dot(r2.T, r1)
                t_dif = np.dot(r2.T, t1 - t2)
                #t_dif = t1 - t2
            else:
                r3, _ = cv.Rodrigues(np.array(c.rvecs[0], dtype=np.float64))
                t3 = np.array(c.tvecs[0], dtype=np.float64)
                rvec_dic[camera] =  np.dot(r3, r_dif)
                tvec_dic[camera] = t3 + np.dot(r3, t_dif) 
                bool_dic[camera] = True

    #rotate_list = ['r0to90','r90to180','r180to270','r270to360']
    """
    rotate_rvec_dic = {'camera1':[],'camera2':[],'camera3':[],'camera4':[],'camera5':[],'camera6':[],'camera7':[],'camera8':[]}
    rotate_tvec_dic = {'camera1':[],'camera2':[],'camera3':[],'camera4':[],'camera5':[],'camera6':[],'camera7':[],'camera8':[]}
    """
    for i in list(bool_dic.keys()):
        camera = i
        folder_path = glob.glob(img_path + "rotation/" + camera + "/*")
        folder_path.sort()
        for j in folder_path[:]:
            file_path = glob.glob(j + '/*.JPG')
            file_path.sort()
            for num, k in enumerate(file_path):
                c = CB_calbration(k,4,3,15)
                c.mtx = np.array(f[camera]['mtx'], dtype=np.float64)
                c.dist = np.array(f[camera]['dist'], dtype=np.float64)
                ret = c.find_CB_coners(4)
                if not(ret):
                    ret = c.find_CB_coners(8)
                #ret = c.find_CB_coners(4, True)
                if ret:
                    c.cal_extrinsic_parameters()
                    c.eval_distortion()
                    r, _ = cv.Rodrigues(np.array(c.rvecs[0], dtype=np.float64))
                    t = np.array(c.tvecs[0], dtype=np.float64)
                
                if j[-1] == '1':
                    if num == 0:
                        r_ref = rvec_dic[camera]
                        t_ref = tvec_dic[camera]
                        r_dif = np.dot(r.T, r_ref)
                        t_dif = np.dot(r.T, t_ref - t)
                        r_c = r_ref
                        t_c = t_ref
                    else:
                        r_c = np.dot(r, r_dif) 
                        t_c = t + np.dot(r, t_dif)
                        if num == len(file_path)-1:
                            r_ref = r_c
                            t_ref = t_c
                else:
                    if num == 0:
                        r_dif = np.dot(r.T, r_ref)
                        t_dif = np.dot(r.T, t_ref - t)
                        r_c = r_ref
                        t_c = t_ref
                    else:
                        r_c = np.dot(r, r_dif) 
                        t_c = t + np.dot(r, t_dif)
                        if num == len(file_path)-1:
                            r_ref = r_c
                            t_ref = t_c
                if num != len(file_path) - 1:
                    rotate_rvec_dic[camera].append(r_c)
                    rotate_tvec_dic[camera].append(t_c)
    """            
    #camera pose visualization

    c_rvec_dic = {'camera1':[],'camera2':[],'camera3':[],'camera4':[],'camera5':[],'camera6':[],'camera7':[],'camera8':[]}
    c_tvec_dic = {'camera1':[],'camera2':[],'camera3':[],'camera4':[],'camera5':[],'camera6':[],'camera7':[],'camera8':[]}

    for i in range(1,9):
        grp = 'camera'+str(i)
        rotate_rvec_dic[grp] = f[grp]['rvec']
        rotate_tvec_dic[grp] = f[grp]['tvec']

    for i in list(bool_dic.keys()):       
        for j in range(len(rotate_rvec_dic[i])):
        #for j in range(10):
            c_rvec_dic[i].append(rotate_rvec_dic[i][j])
            c_tvec_dic[i].append(rotate_tvec_dic[i][j])
       
   
    #from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
    #visualizer = CameraPoseVisualizer([-400, 400], [-400, 400], [-400, 400])
    #visualizer = CameraPoseVisualizer([-1, 1], [-1, 1], [-1, 1])
    #visualizer = CameraPoseVisualizer([-60, 60], [-60, 60], [-60, 60])
    from camera_visualization import camera_visualizer
    color_dic = {'camera1':'red','camera2':'orange','camera3':'yellow','camera4':'lawngreen','camera5':'cyan','camera6':'blue','camera7':'blueviolet','camera8':'magenta'}
    visualizer = camera_visualizer()
    poses = np.zeros((len(c_rvec_dic['camera1']),4,4))
    
    for i in list(bool_dic.keys()):
        index_count = 0 
        for j in range(len(c_rvec_dic[i])):
        #for j in range(1):
            mat = np.concatenate([c_rvec_dic[i][j], c_tvec_dic[i][j]],  axis=1)
            tmp = np.array([0,0,0,1])
            """
            r = mat[0:3,0:3]
            t = mat[0:3,3]
            r_n = r.T
            t_n = -r.T @ t
            mat2 = np.concatenate([r_n, t_n.reshape(3,1)],  axis=1)
            mat2 = np.vstack((mat2, tmp.T))
            """
            mat4 = np.vstack((mat, tmp.T))

            #mat4 = np.concatenate([mat4[0:1, :], -mat4[1:2, :], -mat4[2:3, :], mat4[3:4, :]], 0)
            mat4 = np.linalg.inv(mat4)
            #mat4 = np.concatenate([mat4[0:1, :], -mat4[1:2, :], -mat4[2:3, :], mat4[3:4, :]], 0)
            
            #mat4 = np.concatenate([mat4[0:1, :], -mat4[2:3, :], -mat4[1:2, :], mat4[3:4, :]], 0)
            poses[index_count] = mat4
            index_count += 1
            #visualizer.extrinsic2pyramid(mat4, color_dic[i], 20)
            #visualizer.extrinsic2pyramid(mat4, color_dic[i], 5)
        visualizer.plot_camera_scene(poses,15,color_dic[i],i,True)
    visualizer.show()
    #visualizer.save("PG_2_true.png")
    """
    for i in range(1,9):
        grp = 'camera'+str(i)
        f[grp].create_dataset('rvec', data=rotate_rvec_dic[grp])
        f[grp].create_dataset('tvec', data=rotate_tvec_dic[grp])
    """
    #p = rvec_dic["camera1"]
    #p_, _ = cv.Rodrigues(p)

    f.close()
