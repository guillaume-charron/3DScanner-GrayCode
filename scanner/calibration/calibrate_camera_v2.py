import glob
import os
import cv2
import numpy as np

from scanner.acquisition import Camera

def detect_markers(frame, gray, k, dist, dictionary, params, draw=True):
    rvec = None
    tvec = None
    ret = False
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    # Detect markers and corners
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    allCorners = []
    allIds = []
    if corners is not None and len(corners) > 0:
        if draw:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for corner in corners:
            cv2.cornerSubPix(gray, corner, (3,3), (-1,-1), criteria)
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
        if charuco_ids is not None and len(charuco_ids) > 0:
            if draw:
                frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
            if cam_mtx is not None and cam_dist is not None:
                ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, charuco_board, k, dist, None, None)

                if ret and draw:
                    frame = cv2.drawFrameAxes(frame, k, dist, rvec, tvec, 0.1)

            allCorners = charuco_corners
            allIds = charuco_ids

    return ret, frame, allCorners, allIds

# Define the charuco board parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
charuco_board = cv2.aruco.CharucoBoard((5, 7), 0.04, .02, dictionary)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

# Get previous calibration data for the camera
result_folder = './data/calib_results/cam_1440/'
cam_mtx = None
cam_dist = None
if os.path.exists(os.path.join(result_folder,'cam_mtx.npy')):
    cam_mtx = np.load(os.path.join(result_folder,'cam_mtx.npy'))

if os.path.exists(os.path.join(result_folder,'cam_dist.npy')):
    cam_dist = np.load(os.path.join(result_folder,'cam_dist.npy'))

# Create folder to save calibration images
calibration_folder = './data/CalibrationImgs/camera_1440/'
if not os.path.exists(calibration_folder):
    os.makedirs(calibration_folder)

allCorners = []
allIds = []

width, height = (2560, 1440)

remove_dist = False

if __name__ == '__main__':
    cam = Camera(0, width=width, height=height)
    consecutive_frames = 0
    while True:
        frame = cam.get_frame()

        if frame is not None:
            # Reset parameters
            og_frame = frame.copy()

            # Convert to gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers and corners
            ret, frame, corners, ids = detect_markers(frame, gray, cam_mtx, cam_dist, dictionary, params)

            if cam_mtx is not None and cam_dist is not None and remove_dist:
            # Undistort image
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, cam_dist, (width,height), 1, (width,height))
                dst = cv2.undistort(frame, cam_mtx, cam_dist, None, newcameramtx)
                # crop the image
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]
                resized_frame = cv2.resize(dst, (960, 540))
            else:
                resized_frame = cv2.resize(frame, (960, 540))
            cv2.imshow('frame', resized_frame)

        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            cam.stop_cam()
            break
        if keyPressed == ord('d'):
            remove_dist = not remove_dist
        if keyPressed == ord('c'):
            if ret is not None:
                n = 0
                while os.path.exists(os.path.join(calibration_folder,f'calibrate_{n}.jpg')):
                    n += 1
                cv2.imwrite(os.path.join(calibration_folder,f'calibrate_{n}.jpg'), og_frame)
                allCorners.append(corners)
                allIds.append(ids)
        if keyPressed == ord('k'):
            allCorners.clear()
            allIds.clear()
            # Calibrate projector
            images = glob.glob(os.path.join(calibration_folder,'*.jpg'))
            for fname in images:
                img = cv2.imread(fname)
                print('Reading image:', fname)
                # Convert to gray
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect markers and corners
                ret, frame, corners, ids = detect_markers(frame, gray, cam_mtx, cam_dist, dictionary, params, draw=False)

                if ret:
                    allCorners.append(corners)
                    allIds.append(ids)
                else:
                    print('Bad image:', fname)

            print('Camera calibration')
            cameraMatrixInit = np.array([[ 1000.,    0., 1920/2.],
                                 [    0., 1000., 1080/2.],
                                 [    0.,    0.,           1.]])

            distCoeffsInit = np.zeros((5,1))
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO)
            ret, cam_mtx, cam_dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(allCorners, allIds, charuco_board, (1920,1080), cameraMatrixInit, distCoeffsInit, flags = flags, criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
            print(cam_mtx)
            print(cam_dist)
            print('Error', ret)

            # Save calibration results
            np.save(os.path.join(result_folder, 'cam_mtx.npy'), cam_mtx)
            np.save(os.path.join(result_folder, 'cam_dist.npy'), cam_dist)


