import glob
import os
import cv2 as cv
import numpy as np


class CB:
    def __init__(self, img_path, c, r, cb_size):
        # termination criteria
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.c = c
        self.r = r
        # square size in mm
        self.cb_size = cb_size
        # self.screen = screeninfo.get_monitors()[0]
        _objp = np.zeros((self.c * self.r, 3), np.float32)
        _objp[:, :2] = np.mgrid[0 : self.c, 0 : self.r].T.reshape(-1, 2)
        self.objp = _objp * self.cb_size
        self.objp2 = np.ones((self.c * self.r, 4), np.float32)
        self.objp2[:, :3] = self.objp
        self.w = None
        self.h = None
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.objpoints2 = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        if os.path.isdir(img_path):
            self.images = glob.glob(img_path + "/*.JPG")
            self.images.sort()
        elif os.path.isfile(img_path):
            self.images = [img_path]

        self.img_out_path = img_path + "_c/"
        self.img_size = 0
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = []
        self.tvecs = []
        self.imgpoints_undistorted = []

    def set_intrinsic_parameters(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist

    def flip_corners(self):
        for i in range(len(self.imgpoints)):
            self.imgpoints[i] = np.flip(self.imgpoints[i], 0)

    def find_CB_coners(self, scale, imshow_bool=False):
        count = 0
        for fname in self.images:
            img = cv.imread(fname)
            self.h, self.w = img.shape[:2]
            # img_ = cv.medianBlur(img,5)
            # img_ = cv.GaussianBlur(img,(5,5),0)
            # img_ = cv.bilateralFilter(img,9,75,75)
            img_resized = cv.resize(
                img,
                (img.shape[1] // scale, img.shape[0] // scale),
                interpolation=cv.INTER_AREA,
            )
            # img_resized = cv.resize(img, (img.shape[1]//scale,img.shape[0]//scale), interpolation=cv.INTER_LANCZOS4)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            self.img_size = gray.shape
            gray_resized = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(
                gray_resized, (self.c, self.r), None
            )
            # If found, add object points, image points (after refining them)
            if ret == True:
                print(fname)
                count += 1
                self.objpoints.append(self.objp)
                self.objpoints2.append(self.objp2)
                corners2 = cv.cornerSubPix(
                    gray, corners * scale, (11, 11), (-1, -1), self.criteria
                )
                self.imgpoints.append(corners2)
                # Draw and display the corners
                if imshow_bool:
                    cv.drawChessboardCorners(img, (self.c, self.r), corners2, ret)
                    cv.namedWindow("img", cv.WINDOW_NORMAL)
                    cv.resizeWindow("img", self.screen.width, self.screen.height)
                    cv.imshow("img", img)
                    cv.waitKey(0)
        if imshow_bool:
            cv.destroyAllWindows()
        print(str(count) + " out of " + str(len(self.images)))
        return ret

    def drawAxes(self):
        count = 0
        for i, fname in enumerate(self.images):
            img = cv.imread(fname)
            cv.drawFrameAxes(
                img,
                self.mtx,
                self.dist,
                self.rvecs[i],
                self.tvecs[i],
                length=10,
                thickness=20,
            )
            cv.namedWindow("img", cv.WINDOW_NORMAL)
            cv.resizeWindow("img", self.screen.width, self.screen.height)
            cv.imshow("img", img)
            cv.waitKey(0)

    def cal_camera_parameters(self):
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(
            self.objpoints, self.imgpoints, self.img_size[::-1], None, None
        )

    def cal_extrinsic_parameters(self):
        self.ret, rvecs, tvecs = cv.solvePnP(
            np.array(self.objpoints, dtype=np.float64),
            np.reshape(
                np.array(self.imgpoints, dtype=np.float64), (1, self.c * self.r, 2)
            ),
            self.mtx,
            self.dist,
        )
        self.rvecs.append(rvecs)
        self.tvecs.append(tvecs)

    def undistortion(self, imshow_bool=False):
        for fname in self.images:
            img = cv.imread(fname)
            h, w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (w, h), 1, (w, h)
            )
            dst = cv.undistort(img, self.mtx, self.dist, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y : y + h, x : x + w]
            cv.imwrite(self.img_out_path + os.path.basename(fname), dst)
            if imshow_bool:
                cv.namedWindow("img", cv.WINDOW_NORMAL)
                cv.resizeWindow("img", self.screen.width, self.screen.height)
                cv.imshow("img", dst)
                cv.waitKey(0)
        if imshow_bool:
            cv.destroyAllWindows()

    def undistortPoint(self):
        for i in range(len(self.imgpoints)):
            temp = cv.undistortImagePoints(self.imgpoints[i], self.mtx, self.dist)
            self.imgpoints_undistorted.append(temp.flatten().reshape(-1, 2))

    def eval_distortion(self):
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist
            )
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error
        print("total error: {}".format(mean_error / len(self.objpoints)))
        return mean_error / len(self.objpoints)

    def show_distortion_plot(self):
        pts_new, _ = cv.projectPoints(
            self.objpoints[0], self.rvecs[0], self.tvecs[0], self.mtx, self.dist
        )
        pts, _ = cv.projectPoints(
            self.objpoints[0], self.rvecs[0], self.tvecs[0], self.mtx, None
        )
        with open(self.img_out_path + "distortion_sample.txt", "w") as f:
            for i in range(pts.shape[0]):
                f.write(
                    "%f,%f,%f,%f"
                    % (pts[i, 0, 0], pts[i, 0, 1], pts_new[i, 0, 0], pts_new[i, 0, 1])
                )
                f.write("\n")

    def show_distortion_plot1(self):
        pts_new, _ = cv.projectPoints(
            self.objpoints[0], self.rvecs[0], self.tvecs[0], self.mtx, self.dist
        )
