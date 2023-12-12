import glob
import os
import cv2
import numpy as np

from scanner.acquisition import Camera

def remove_dist(img, k, dist):
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(k, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, k, dist, None, newcameramtx)
    x, y, w, h = roi
    return dst[y:y+h, x:x+w]

def detect_markers(frame, gray, k, dist, dictionary, params, draw=True):
    rvec = None
    tvec = None
    ret = False
    H = None
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    # Detect markers and corners
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    if corners is not None and len(corners) > 0:
        if draw:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for corner in corners:
            cv2.cornerSubPix(gray, corner, (3,3), (-1,-1), criteria)

        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
        if charuco_ids is not None and len(charuco_ids) > 0:
            if draw:
                frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
            ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, charuco_board, k, dist, None, None)

            if ret and draw:
                frame = cv2.drawFrameAxes(frame, k, dist, rvec, tvec, 0.1)

            obj_points = np.reshape(np.array(charuco_board.getObjPoints())[ids][:,:,:, :2], (len(corners)*4, 2))
            corners_flatten = np.reshape(corners, (len(corners)*4, 2))
            H, _ = cv2.findHomography(corners_flatten, obj_points, cv2.RANSAC, 5.0)

    return ret, frame, rvec, tvec, H

def detect_circle_grid(frame, gray, k, dist, shape, rvec, tvec, H, draw=True):
    ret, circles = cv2.findCirclesGrid(gray, shape, flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)

    circles3D = None
    if ret:
        if draw:
            frame = cv2.drawChessboardCorners(frame, shape, circles, ret)

        circles3D = cv2.perspectiveTransform(circles, H)
        circles3D = np.pad(circles3D, ((0,0), (0,0), (0,1)), 'constant', constant_values=0)

    return ret, frame, circles, circles3D.astype(np.float32) if circles3D is not None else None

def build_circle_grid_pts(nb_col, nb_row, circle_r):
    circle_2d_pts = np.zeros((nb_col*nb_row, 2), dtype=np.int32)
    count = 0
    for i in range(nb_row-1, -1, -1):
        for j in range(nb_col-1, -1, -1):
            if i % 2 == 0:
                pos_x = j * 6 * circle_r + (3 * circle_r)
            else:
                pos_x = j * 6 * circle_r
            pos_y = i * 3 * circle_r
            circle_2d_pts[count] = [pos_x, pos_y]
            count += 1
    return circle_2d_pts

# Define the charuco board parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
charuco_board = cv2.aruco.CharucoBoard((5, 7), 0.04, .02, dictionary)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

# Get previous calibration data for the camera
cam_mtx = np.load('data/calib_results/cam_mtx.npy')
cam_dist = np.load('data/calib_results/cam_dist.npy')

# Get previous calibration data for the camera
proj_mtx = None
proj_dist = None
if os.path.exists('./data/calib_results/proj_mtx.npy') and os.path.exists('./data/calib_results/proj_dist.npy'):
    proj_mtx = np.load('./data/calib_results/proj_mtx.npy')
    proj_dist = np.load('./data/calib_results/proj_dist.npy')

# Create empty image to draw circles
img_circle = np.zeros((1080,1920,4), dtype=np.uint8)

# Create folder to save calibration images
calibration_folder = './data/CalibrationImgs/projector/'
if not os.path.exists(calibration_folder):
    os.makedirs(calibration_folder)

# Define circle grid points in 2D
nb_col = 4
nb_row = 11
circle_r = 15
default_x, default_y = (800, 350)

circle_2d_pts = build_circle_grid_pts(nb_col, nb_row, circle_r)

proj_obj_pts = []
proj_circle_pts = []
cam_circle_pts = []

if __name__ == '__main__':
    cam = Camera(0)
    consecutive_frames = 0
    while True:
        frame = cam.get_frame()

        if frame is not None:
            # Reset parameters
            og_frame = frame.copy()
            circles_3d = None

            # Convert to gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers and corners
            ret, frame, rvec, tvec, H = detect_markers(frame, gray, cam_mtx, cam_dist, dictionary, params)

            if ret:
                # Find projected circles
                ret, frame, circles_2d_cam, circles_3d = detect_circle_grid(frame, gray, cam_mtx, cam_dist, (nb_col, nb_row), rvec, tvec, H)

            if ret:
                consecutive_frames += 1
            else:
                consecutive_frames = 0

            if consecutive_frames > 5:
                n = 0
                while os.path.exists(os.path.join(calibration_folder,f'calibrate_{n}.jpg')):
                    n += 1
                cv2.imwrite(os.path.join(calibration_folder,f'calibrate_{n}.jpg'), og_frame)
                proj_obj_pts.append(circles_3d)
                proj_circle_pts.append(circle_2d)
                print('Saved calibration image')
                consecutive_frames = 0


            proj_img = img_circle.copy()
            # if proj_mtx is not None and proj_dist is not None and rvec is not None and tvec is not None:

            circle_2d = circle_2d_pts + [default_x, default_y] # TODO adjust to follow charuco board

            white_padding = 3*circle_r
            min_x = np.min(circle_2d[:,0]) - white_padding
            max_x = np.max(circle_2d[:,0]) + white_padding
            min_y = np.min(circle_2d[:,1]) - white_padding
            max_y = np.max(circle_2d[:,1]) + white_padding
            proj_img[min_y:max_y, min_x:max_x] = cv2.bitwise_not(proj_img[min_y:max_y, min_x:max_x])
            for c in circle_2d:
                proj_img = cv2.circle(proj_img, tuple(c.astype(np.int32)), circle_r, (0,0,0), cv2.FILLED)
            resized_frame = cv2.resize(frame, (960, 540))
            cv2.imshow('frame', resized_frame)
            cv2.imshow('circle', proj_img)

        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            cam.stop_cam()
            break
        if keyPressed == ord('c'):
            if circles_3d is not None:
                n = 0
                while os.path.exists(os.path.join(calibration_folder,f'calibrate_{n}.jpg')):
                    n += 1
                cv2.imwrite(os.path.join(calibration_folder,f'calibrate_{n}.jpg'), og_frame)
                proj_obj_pts.append(circles_3d)
                proj_circle_pts.append(circle_2d)
        if keyPressed == ord('k'):
            proj_obj_pts.clear()
            proj_circle_pts.clear()
            cam_circle_pts.clear()
            # Calibrate projector
            images = glob.glob(os.path.join(calibration_folder,'*.jpg'))
            for fname in images:
                img = cv2.imread(fname)
                # Convert to gray
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect markers and corners
                ret, img, rvec, tvec, H = detect_markers(img, gray, cam_mtx, cam_dist, dictionary, params, draw=False)

                if ret:
                    # Find projected circles
                    ret, img, circles_2d_cam, circles_3d = detect_circle_grid(img, gray, cam_mtx, cam_dist, (nb_col, nb_row), rvec, tvec, H, draw=False)
                    if ret:
                        proj_obj_pts.append(circles_3d)
                        proj_circle_pts.append(np.expand_dims(circle_2d.astype(np.float32), axis=1))
                        cam_circle_pts.append(circles_2d_cam)
                    else:
                        print('Bad image:', fname)

            # print('Camera calibration')
            # ret, cam_mtx, cam_dist, rvecs, tvecs = cv2.calibrateCamera(proj_obj_pts, cam_circle_pts, (1920,1080), None, None)
            # print(cam_mtx)
            # print(cam_dist)
            # print('Error', ret)

            print('\nProjector calibration')
            ret, proj_mtx, proj_dist, rvecs, tvecs = cv2.calibrateCamera(proj_obj_pts, proj_circle_pts, (1920,1080), None, None)
            print(proj_mtx)
            print(proj_dist)
            print('Error', ret)

            print('\nStereo calibration')
            error, cam_mtx, cam_dist, proj_mtx, proj_dist, proj_R, proj_T,_,_ = cv2.stereoCalibrate(proj_obj_pts, cam_circle_pts, proj_circle_pts, cam_mtx, cam_dist, proj_mtx, proj_dist, (1920,1080), flags=cv2.CALIB_FIX_INTRINSIC) #, flags=cv2.CALIB_USE_INTRINSIC_GUESS)#
            #error, cam_mtx, cam_dist, proj_mtx, proj_dist, proj_R, proj_T,_,_ = cv2.stereoCalibrate(proj_obj_pts, cam_circle_pts, proj_circle_pts, cam_mtx, cam_dist, proj_mtx, proj_dist, (1920,1080), flags=cv2.CALIB_USE_INTRINSIC_GUESS) #, flags=cv2.)#
            print('Camera parameters')
            print(cam_mtx, cam_dist)
            print('Projector parameters')
            print(proj_mtx, proj_dist)
            print('Rotation and translation')
            print(proj_R)
            print(proj_T)
            print('Error', error)

            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cam_mtx, cam_dist, proj_mtx, proj_dist, (1920,1080), proj_R, proj_T)

            # Save calibration results
            np.save('./data/proj_mtx.npy', proj_mtx)
            np.save('./data/proj_dist.npy', proj_dist)
            np.save('./data/R.npy', proj_R)
            np.save('./data/T.npy', proj_T)
            np.save('./data/R1.npy', R1)
            np.save('./data/R2.npy', R2)
            np.save('./data/P1.npy', P1)
            np.save('./data/P2.npy', P2)


