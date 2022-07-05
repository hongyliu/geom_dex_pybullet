import glob
import pybullet as p
import time
import numpy as np
import pybullet_data
import cv2
import json
from shadowhand_gym.envs.camera import CameraArray, Camera, CalibratedCamera


def get_calibrate_camera():
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (9, 9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) / 50.0 * 3
    # print(objp[:, :10, :])
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob('./images/chessboard_camera_*.jpg')
    gray_shape = []
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imwrite("./images/corners_camera_{}".format(fname.split('_')[-1]), img)


    cv2.destroyAllWindows()


    """
    Performing camera calibration by
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the
    detected corners (imgpoints)
    """
    calibrate_parameters = []
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    for i, rvec in enumerate(rvecs):
        camera_calibrate = {}
        rotation, _ = cv2.Rodrigues(rvec)
        # print(rotation)
        extrinsic = np.concatenate([np.concatenate([rotation, tvecs[i]], axis=1), np.asarray([[0, 0, 0, 1]])], axis=0)
        # print(extrinsic)
        camera_calibrate['class_name'] = "PinholeCameraParameters"
        camera_calibrate['extrinsic'] = extrinsic.transpose().flatten().tolist()
        camera_calibrate['distortion'] = dist.tolist()
        camera_calibrate['intrinsic'] = {
            'height': gray_shape[1],
            'intrinsic_matrix': mtx.transpose().flatten().tolist(),
            'width': gray_shape[0]
        }
        camera_calibrate['version_major'] = 1
        camera_calibrate['version_minor'] = 0
        calibrate_parameters.append(camera_calibrate)
    calibrate_result = {
        "class_name": "PinholeCameraTrajectory",
        "parameters": calibrate_parameters,
        "version_major": 1,
        "version_minor": 0
    }

    with open("./camera.json", 'w') as file:
        json.dump(calibrate_result, file, indent=4)


def get_chess_img():
    physicsClient = p.connect(p.DIRECT, options=(
        "--background_color_red=255 "
        "--background_color_green=255 "
        "--background_color_blue=255"
    ))  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.configureDebugVisualizer(lightPosition=[0, 0, 9])

    chessId = p.loadURDF("table_square/table_square.urdf", (0, 0, -0.68))
    camera_list = []
    camera_list.append(Camera((0.2, 0.2, 0.7), (0.707, 0.691, -0.147, -0.043)))
    camera_list.append(Camera((0.2, 0.1, 0.7), (0.707, 0.691, -0.147, -0.043)))
    camera_list.append(Camera((0.2, 0, 0.7), (0.755, 0.642, -0.119, -0.063)))
    camera_list.append(Camera((0.2, -0.1, 0.7), (0.791, 0.593, -0.057, -0.136)))
    camera_list.append(Camera((0.2, -0.2, 0.7), (0.791, 0.593, -0.057, -0.136)))
    cameras = CameraArray(camera_list)
    images, _ = cameras.get_images()
    for i, img in enumerate(images):
        cv2.imwrite("./images/chessboard_camera_{}.jpg".format(i), img)
    # set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    # for i in range(10000):
    #     p.stepSimulation()
    #     time.sleep(1. / 240.)

    p.disconnect()


if __name__ == '__main__':
    get_chess_img()
    get_calibrate_camera()
