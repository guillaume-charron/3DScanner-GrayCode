import glob
import os
import cv2
import numpy as np

from acquisition.threaded_film_pattern import Camera


def fromCamToWorld(cameraMatrix, distCoeff, rV, tV, imgPoints):
    s = len(imgPoints)
    invK = np.linalg.inv(cameraMatrix)
    #imgPoints = cv2.convertPointsToHomogeneous(cv2.undistortPoints(imgPoints, cameraMatrix, distCoeff))

    r = rV.astype(np.float32)
    t = tV.astype(np.float32)
    rMat, _ = cv2.Rodrigues(r)
    transPlaneToCam = np.linalg.inv(rMat) @ t
    world_points = []
    for i in range(s):
        wpTemp = []
        s2 = len(imgPoints[i])
        for j in range(s2):
            coords = np.array([[imgPoints[i][j][0]], [imgPoints[i][j][1]], [1.0]], dtype=np.float32)
            
            worldPtCam = invK @ coords
            worldPtPlane = np.linalg.inv(rMat) @ worldPtCam

            scale = (transPlaneToCam[2] / worldPtPlane[2])[0]
            worldPtPlaneReproject = scale * worldPtPlane - transPlaneToCam
            

            pt = np.array([worldPtPlaneReproject[0][0], worldPtPlaneReproject[1][0], 0], dtype=np.float32)
            wpTemp.append(pt)
        world_points.append(wpTemp)

    return np.array(world_points, dtype=np.float32)

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

    # Detect markers and corners
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    if corners is not None and len(corners) > 0:
        if draw:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, charuco_board)
        if charuco_ids is not None and len(charuco_ids) > 0:
            if draw:
                frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
            ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, charuco_board, k, dist, None, None)
            if ret and draw:
                frame = cv2.drawFrameAxes(frame, k, dist, rvec, tvec, 0.1)

    return ret, frame, rvec, tvec

def detect_circle_grid(frame, gray, k, dist, shape, rvec, tvec, draw=True):
    ret, circles = cv2.findCirclesGrid(gray, shape, flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
    circles3D_reprojected_on_board = None
    world_pts = None
    circles3D = None
    if ret:
        if draw:
            frame = cv2.drawChessboardCorners(frame, shape, circles, ret)
        # ray-plane intersection: circle-center to chessboard-plane
        circles3D = intersectCirclesRaysToBoard(circles, rvec, tvec, k, dist)
        for i in range(len(circles3D)):
            circles3D[i] = circles3D[i] / circles3D[i][2]
            circles3D[i][2] = 0

        world_pts = fromCamToWorld(k, dist, rvec, tvec, circles)

        # re-project on camera for verification
        circles3D_reprojected, _ = cv2.projectPoints(circles3D, (0,0,0), (0,0,0), k, dist)

        # # Project on board
        # circles3D_reprojected_on_board, _ = cv2.projectPoints(circles3D, rvec, tvec, k, dist)
        # circles3D_reprojected_on_board = np.pad(circles3D_reprojected_on_board, ((0,0), (0,0), (0, 1)))
        if draw:
            for c in circles3D_reprojected:
                frame = cv2.circle(frame, tuple(c.astype(np.int32)[0]), 5, (255,255,0), cv2.FILLED)
    return ret, frame, circles, circles3D.astype(np.float32) if circles3D is not None else None


# Define the charuco board parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
charuco_board = cv2.aruco.CharucoBoard((5, 7), 0.04, .02, dictionary)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
#cv2.imwrite('charuco.jpg', charuco_board.generateImage((1000, 1000), 5, 1))

# Get previous calibration data for the camera
cam_mtx = np.load('data/mtx.npy')
cam_dist = np.load('data/dist.npy')

# Get previous calibration data for the camera
proj_mtx = None
proj_dist = None
if os.path.exists('./data/proj_mtx.npy') and os.path.exists('./data/proj_dist.npy'):
    proj_mtx = np.load('./data/proj_mtx.npy')
    proj_dist = np.load('./data/proj_dist.npy')

# Load circle grid image
img_circle = np.zeros((1080,1920,4), dtype=np.uint8)
#img_circle = cv2.bitwise_not(img_circle) # TODO remove when projected (black = no background when projected)

# Create folder to save calibration images
calibration_folder = './calibration/CalibrationImgs/projector/'
if not os.path.exists(calibration_folder):
    os.makedirs(calibration_folder)

# Define circle grid points in 2D
nb_col = 4
nb_row = 11
circle_r = 20
default_x, default_y = (800, 400)

circle_2d_pts = np.zeros((nb_col*nb_row, 2), dtype=np.int32)
count = 0
for i in range(nb_row):
    for j in range(nb_col-1, -1, -1):
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
            ret, frame, rvec, tvec = detect_markers(frame, gray, cam_mtx, cam_dist, dictionary, params)

            if ret:
                # Find projected circles
                ret, frame, circles_2d_cam, circles_3d = detect_circle_grid(frame, gray, cam_mtx, cam_dist, (nb_col, nb_row), rvec, tvec)
            
            if ret:
                consecutive_frames += 1
            else:
                consecutive_frames = 0
            
            if consecutive_frames > 10:
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
            cv2.imshow('frame', frame)
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
            # Calibrate projector
            images = glob.glob(os.path.join(calibration_folder,'*.jpg'))
            for fname in images:
                img = cv2.imread(fname)
                # Convert to gray
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect markers and corners
                ret, img, rvec, tvec = detect_markers(img, gray, cam_mtx, cam_dist, dictionary, params, draw=False)

                if ret:
                    # Find projected circles
                    ret, img, circles_2d_cam, circles_3d = detect_circle_grid(img, gray, cam_mtx, cam_dist, (nb_col, nb_row), rvec, tvec, draw=False)
                    proj_obj_pts.append(circles_3d)
                    proj_circle_pts.append(circle_2d.astype(np.float32))

            ret, proj_mtx, proj_dist, rvecs, tvecs = cv2.calibrateCamera(proj_obj_pts, proj_circle_pts, (1920,1080), None, None)
            print(proj_mtx)
            print(proj_dist)
            np.save('./data/proj_mtx.npy', proj_mtx)
            np.save('./data/proj_dist.npy', proj_dist)

