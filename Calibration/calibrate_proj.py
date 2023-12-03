import os
import cv2
import numpy as np

from acquisition.threaded_film_pattern import Camera

# Took from https://www.morethantechnical.com/blog/2017/11/17/projector-camera-calibration-the-easy-way/
def intersectCirclesRaysToBoard(circles, rvec, t, K, dist_coef):
    circles_normalized = cv2.convertPointsToHomogeneous(cv2.undistortPoints(circles, K, dist_coef))
    if not rvec.size:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
    plane_normal = R[2,:] # last row of plane rotation matrix is normal to plane
    plane_point = t.T     # t is a point on the plane
    epsilon = 1e-06
    circles_3d = np.zeros((0,3), dtype=np.float32)
    for p in circles_normalized:
        ray_direction = p / np.linalg.norm(p)
        ray_point = p
        ndotu = plane_normal.dot(ray_direction.T)
        if abs(ndotu) < epsilon:
            print ("no intersection or line is within plane")
        w = ray_point - plane_point
        si = -plane_normal.dot(w.T) / ndotu
        Psi = w + si * ray_direction + plane_point
        circles_3d = np.append(circles_3d, Psi, axis = 0)
    return circles_3d

# Define the charuco board parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
charuco_board = cv2.aruco.CharucoBoard((5, 7), 0.04, .02, dictionary)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
#cv2.imwrite('charuco.jpg', charuco_board.generateImage((200*3, 200*3), 10, 1))

# Get previous calibration data for the camera
cam_mtx = np.load('data/mtx.npy')
cam_dist = np.load('data/dist.npy')

# Load circle grid image
img_circle = np.zeros((1080,1920,4), dtype=np.uint8)
img_circle = cv2.bitwise_not(img_circle) # TODO remove when projected (black = no background when projected)

# Create folder to save calibration images
calibration_folder = './calibration/CalibrationImgs/projector/'
if not os.path.exists(calibration_folder):
    os.mkdir(calibration_folder)

# Define circle grid points in 2D
nb_col = 4
nb_row = 11
circle_r = 10
default_x, default_y = (800, 400)

circle_2d_pts = np.zeros((nb_col*nb_row, 2), dtype=np.int32)
count = 0
for i in range(nb_row):
    for j in range(nb_col):
        if i % 2 == 0:
            pos_x = j * 6 * circle_r + (3 * circle_r)
        else:
            pos_x = j * 6 * circle_r 
        pos_y = i * 3 * circle_r
        circle_2d_pts[count] = [pos_x, pos_y]
        count += 1

proj_obj_pts = []
proj_circle_pts = []

if __name__ == '__main__':
    cam = Camera()
    while True:
        frame = cam.get_frame()

        if frame is not None:
            # Reset parameters
            og_frame = frame.copy()
            circles3D = None

            # Remove distortion from the image
            h,  w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, cam_dist, (w,h), 1, (w,h))
            dst = cv2.undistort(frame, cam_mtx, cam_dist, None, newcameramtx)
            x, y, w, h = roi
            frame = dst[y:y+h, x:x+w]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers and corners
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

            if corners is not None and len(corners) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, charuco_board)
                if charuco_ids is not None and len(charuco_ids) > 0:
                    #print(charuco_corners.shape, charuco_ids.shape)
                    cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
                    ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, charuco_board, cam_mtx, cam_dist, None, None)
                    if ret:
                        #cv2.drawFrameAxes(frame, cam_mtx, cam_dist, rvec, tvec, 0.1)

                        # Find projected circles
                        ret, circles = cv2.findCirclesGrid(gray, (nb_col, nb_row), flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
                        if ret:
                            frame = cv2.drawChessboardCorners(frame, (nb_col, nb_row), circles, ret)
                            # ray-plane intersection: circle-center to chessboard-plane
                            circles3D = intersectCirclesRaysToBoard(circles, rvec, tvec, cam_mtx, cam_dist)
                            # re-project on camera for verification
                            circles3D_reprojected, _ = cv2.projectPoints(circles3D, (0,0,0), (0,0,0), cam_mtx, cam_dist)
                            for c in circles3D_reprojected:
                                cv2.circle(frame, tuple(c.astype(np.int32)[0]), 5, (255,255,0), cv2.FILLED)

            

            proj_img = img_circle.copy()
            circle_2d = circle_2d_pts + [default_x, default_y] # TODO adjust to follow charuco board
            for c in circle_2d:
                proj_img = cv2.circle(proj_img, tuple(c.astype(np.int32)), circle_r, (255,255,0), cv2.FILLED)
            cv2.imshow('frame', frame)
            cv2.imshow('circle', proj_img)
        
        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            cam.stop_cam()
            break
        if keyPressed == ord('c'):
            if circles3D is not None:
                n = 0
                while os.path.exists(os.path.join(calibration_folder,f'calibrate_{n}.jpg')):
                    n += 1
                cv2.imwrite(os.path.join(calibration_folder,f'calibrate_{n}.jpg'), og_frame)
                proj_obj_pts.append(circles3D)
                proj_circle_pts.append(circle_2d)
        if keyPressed == ord('k'):
            # Calibrate projector
            ret, proj_mtx, proj_dist, rvecs, tvecs = cv2.calibrateCamera(proj_obj_pts, proj_circle_pts, (1920,1080), None, None)
            np.save('./data/proj_mtx.npy', proj_mtx)
            np.save('./data/proj_dist.npy', proj_dist)

