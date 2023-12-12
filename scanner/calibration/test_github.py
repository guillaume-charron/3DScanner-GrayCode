import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import os
import sys
import signal
import math
#from scipy.spatial.transform import Rotation as sp_rot

# Projector parameters
w_proj = 1920
h_proj = 1080
w_cam = 1920
h_cam = 1080
R_proj = np.array([[-0.99076131, -0.00491919, -0.13552799],
                     [ 0.02515179, -0.98866989, -0.14798394],
                     [-0.13326448, -0.15002553,  0.97965959]])
T_proj = np.array([ 0.09225259, -0.25273821,  1.1683442 ])
# Camera parameters
K_cam_origin = np.array([[639.930358887,0,639.150634766],[0,639.930358887,351.240905762],[0,0,1]])
K_cam = K_cam_origin
dist_coef = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
# Text parameters
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 0.5
fontColor              = (255,0,255)
lineType               = 2
# Geometric parameters (manually measured)
board_to_world = np.array([0., 0., 0.2591491014277822])

def plot_charuco(R, tvec, charucoCorners, charucoIds, K=K_cam, dist_coef=dist_coef, ori_idx=0):
    charucoCorners_normalized = cv2.convertPointsToHomogeneous(cv2.undistortPoints(charucoCorners, K, dist_coef))
    charucoCorners_normalized = np.squeeze(charucoCorners_normalized)
    t = np.array(tvec)
    plane_normal = R[2,:] # last row of plane rotation matrix is normal to plane
    plane_point = t.reshape(3,)     # t is a point on the plane
    charucoCorners = []
    epsilon = 1e-06
    for p in charucoCorners_normalized:
        p = p.reshape(3,)
        ray_direction = p / np.linalg.norm(p)
        ray_point = p

        ndotu = plane_normal.dot(ray_direction.T)

        if abs(ndotu) < epsilon:
            print ("no intersection or line is within plane")

        w = ray_point - plane_point
        si = -plane_normal.dot(w.T) / ndotu
        v = w + si * ray_direction
        Psi = w + si * ray_direction + plane_point
        charucoCorners.append(Psi)

    # dist = np.array([np.linalg.norm(tvec - np.array(pt)) for pt in charucoCorners])
    ori = charucoCorners[ori_idx]
    # print('Origin id: ', charucoIds[ori_idx], ' origin xyz: ', ori)
    charuco_plot = []
    # charuco_plot.append(mlab.points3d(ori[0], ori[1], ori[2], scale_factor=0.01, color=(1,0,0)))
    # mlab.text3d(ori[0], ori[1], ori[2], str(charucoIds[ori_idx]), scale=(0.01,0.01,0.01))
    new_corners = np.delete(charucoCorners, (ori_idx), axis=0)
    # for pt in new_corners:
    #     charuco_plot.append(mlab.points3d(pt[0], pt[1], pt[2], scale_factor=0.01, color=(0,0,1)))
    return charucoCorners, np.array(ori)

def get_charuco(frame, K_cam=K_cam, dist_coef=dist_coef):
    # detect charuco
    # K = K_cam.copy()
    K = K_cam.copy()
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    corners = np.array(corners)
    ids = np.array(ids)
    corners = corners[:, 0, :, :]
    ids = ids[:, 0]
    print(corners.shape, ids.shape)
    #corners, ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(frame, cb, corners, ids, rejected, cameraMatrix=K, distCoeffs=dist_coef)
    if corners == None or len(corners) == 0:
        return None
    ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, cb)
    cv2.aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds)
    # cv2.imshow('charuco',frame)
    # cv2.waitKey(0)
    ret, K, dist_coef, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco([charucoCorners],
                                                                       [charucoIds],
                                                                       cb,
                                                                       (1920, 1080),
                                                                       K,
                                                                       dist_coef,
                                                                       #flags = cv2.CALIB_USE_INTRINSIC_GUESS)
                                                                       flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_USE_INTRINSIC_GUESS  + cv2.CALIB_FIX_FOCAL_LENGTH + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_K1  + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6)


    K_cam = K
    return [frame, charucoCorners, charucoIds, rvecs, tvecs]

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    x = np.degrees(x)
    y = np.degrees(y)
    z = np.degrees(z)

    return np.array([x, y, z])

def distBoardToPoint(p, plane_point, plane_normal, plane_x, plane_y, epsilon=1e-06):
    p = p.reshape(3,)
    ray_direction = p / np.linalg.norm(p)
    ray_point = p

    ndotu = plane_normal.dot(ray_direction.T)

    if abs(ndotu) < epsilon:
        print ("no intersection or line is within plane")

    w = ray_point - plane_point
    si = -plane_normal.dot(w.T) / ndotu
    v = w + si * ray_direction
    vx = v.dot(plane_x)
    vy = v.dot(plane_y)
    return np.array([vx, vy, 0])

def intersectCirclesRaysToBoard(circles, rvec, t, K, dist_coef, init_param=False):
    circles_normalized = cv2.convertPointsToHomogeneous(cv2.undistortPoints(circles, K, dist_coef)) # z= 1

    if not rvec.size:
        return None

    R, _ = cv2.Rodrigues(rvec)

    # https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
    plane_x = R[0,:]
    plane_y = R[1,:]
    plane_normal = R[2,:] # last row of plane rotation matrix is normal to plane
    plane_point = t.reshape(3,)     # t is a point on the plane
    epsilon = 1e-06


    if init_param:
        # Find the coordinate of the origin
        R_cam = np.linalg.inv(R)
        s_w = w_cam/(w_proj*1.0)
        s_h = h_cam/(h_proj*1.0)
        origin2D = np.expand_dims(np.expand_dims([s_w * 452.0, s_h * 155.0], 0), 0)
        origin3D = cv2.convertPointsToHomogeneous(cv2.undistortPoints(origin2D, K, dist_coef))
        print("From board center to origin: \n", -1 * distBoardToPoint(np.array(origin3D), plane_point, plane_normal, plane_x, plane_y, epsilon=epsilon))

    circles_2d = []
    circles_3d = []
    for p in circles_normalized:
        p = p.reshape(3,)
        ray_direction = p / np.linalg.norm(p)
        ray_point = p

        ndotu = plane_normal.dot(ray_direction.T)

        if abs(ndotu) < epsilon:
            print ("no intersection or line is within plane")

        w = ray_point - plane_point
        si = -plane_normal.dot(w.T) / ndotu
        v = w + si * ray_direction
        Psi = w + si * ray_direction + plane_point
        vx = v.dot(plane_x)
        vy = v.dot(plane_y)
        circles_2d.append(np.array([vx,vy,0.0]))
        circles_3d.append(Psi)
    return np.array(circles_2d), np.array(circles_3d)

def sort_circles(circles, ori):
    # Find the closest circle to the origin of the board

    print('circles: ', circles, ' ori: ', ori)
    min_dist = 10e10
    min_idx = None
    for i in range(len(circles)):
        circ = circles[i]
        dist = np.sqrt((circ[0] - ori[0])**2 - (circ[1] - ori[1])**2)
        if min_dist > dist:
            min_dist = dist
            min_idx = i
    print('min dist: ', min_dist, ' circle idx: ', min_idx, ' circle coord: ', circles[min_idx])

def get_circle(frame, ret_charuco, ori_idx, charucoCorners, projCirclePoints=None, K_cam=K_cam, dist_coeff=dist_coef, init_param=False):
    frame_charuco, charucoCorners, charucoIds, rvec, tvec = ret_charuco
    # origin = np.squeeze(charucoCorners[ori_idx])
    rvec = np.array(rvec)
    tvec = np.array(tvec)
    K = K_cam.copy()
    dist = dist_coeff.copy()
    # detect circles
    img = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, circles = cv2.findCirclesGrid(gray, circles_grid_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
    # Plot
    cv2.drawChessboardCorners(frame_charuco, circles_grid_size, circles, ret)
    circles_tmp = np.squeeze(circles)
    # sort_circles(circles_tmp, origin)
    # ray-plane intersection: circle-center to chessboard-plane
    circles2D, circles3D = intersectCirclesRaysToBoard(circles, rvec, tvec, K, dist, init_param=init_param)
    # if projCirclePoints is not None:
    #     for i in range(len(circles_tmp)):
    #         circ = circles_tmp[i]
    #         cv2.putText(frame_charuco, str(projCirclePoints[i]), (circ[0],circ[1]), font, fontScale, fontColor, lineType)
        # cv2.imshow('circle with charuco', frame_charuco)
        # cv2.waitKey(0)
    return circles, circles2D, circles3D

# 960, 477
def get_circle_coord(num_sets=1, top_left = [800, 203], h_sep = 137, v_sep = 68, l_sep = 68, test_plot=False):
    # h_diff = 477-203
    # circ = []
    # for y in range(4):
    #     p0 = [top_left[0]+y%2*l_sep, top_left[1]+y*v_sep]
    #     for x in range(4):
    #         p = [p0[0]+x*h_sep, p0[1]]
    #         circ.append(p)
    # for y in range(7):
    #     p0 = [top_left[0]+y%2*l_sep, top_left[1] + h_diff +y*v_sep]
    #     for x in range(4):
    #         p = [p0[0]+x*h_sep, p0[1]]
    #         circ.append(p)
    # circ_ret = [circ for i in range(num_sets)]

    # if test_plot:
    #     img_dots = cv2.imread("img_calib/calibration/capture.png")
    #     img = img_dots.copy()
    #     cnt = 0
    #     for p in circ_ret[0]:
    #         cv2.circle(img, tuple(p), radius=10, thickness=-1, color=(200,0,0))
    #         cv2.putText(img, str(cnt), tuple(p), font, fontScale, fontColor, lineType)
    #         cnt += 1
    #     cv2.imshow("img_dots", img)
    #     cv2.waitKey(0)
    nb_row = 11
    nb_col = 4
    circle_r = 15
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
    circle_2d_pts += [800,350]
    circ_ret = [circle_2d_pts for i in range(num_sets)]
    return np.array(circ_ret)

# create Aruco board
# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
# markerLength = 0.04 # unit: meters
# markerSeparation = 0.02 # unit: meters
# cb = aruco.CharucoBoard_create(5, 7, markerLength, markerSeparation, aruco_dict)
# parameters = aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
cb = cv2.aruco.CharucoBoard((5, 7), 0.04, .02, aruco_dict)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

circles_grid_size = (4,11)
charucoCornersAccum = []
charucoIdsAccum = []
rvecsAccum = []
tvecsAccum = []
circleWorld = []
circleCam = []


# # Init visualizations
# import mayavi
# import mayavi.mlab as mlab

# def plot_vec(tvec, direc, color=(1,1,0)):
#     return mlab.quiver3d(tvec[0],tvec[1],tvec[2],direc[0],direc[1],direc[2], line_width=3, scale_factor=.1, color=color)

# def plot_plane(tvec, R=None, color=(1,0,0)):
#     pt1 = np.array([.1,.1,0], dtype=np.float32)
#     pt2 = np.array([.1,-.1,0], dtype=np.float32)
#     pt3 = np.array([-.1,.1,0], dtype=np.float32)
#     pt4 = np.array([-.1,-.1,0], dtype=np.float32)
#     pts = np.array([pt1,pt2,pt3,pt4])

#     tvec = np.array(tvec).reshape((3))
#     vecs = []
#     if R is not None:
#         pts_rot = [np.matmul(R, p) + np.array(tvec) for p in pts]
#         pts = np.array(pts_rot)
#         for i in range(3):
#             mlab.text3d(tvec[0]+R[i,0]*0.1, tvec[1]+R[i,1]*0.1, tvec[2]+R[i,2]*0.1, str(i), scale=(0.01,0.01,0.01))
#             vec = mlab.quiver3d(tvec[0],tvec[1],tvec[2],R[i,0],R[i,1],R[i,2], line_width=3, scale_factor=.1, color=color)
#             vecs.append(vec)
#     else:
#         mlab.text3d(tvec[0]+1*0.1, tvec[1], tvec[2], 'x', scale=(0.01,0.01,0.01))
#         mlab.text3d(tvec[0], tvec[1]+1*0.1, tvec[2], 'y', scale=(0.01,0.01,0.01))
#         mlab.text3d(tvec[0], tvec[1], tvec[2]+1*0.1, 'z', scale=(0.01,0.01,0.01))
#         vecs = [mlab.quiver3d(tvec[0],tvec[1],tvec[2],1,0,0, line_width=3, scale_factor=.1, color=color),
#                 mlab.quiver3d(tvec[0],tvec[1],tvec[2],0,1,0, line_width=3, scale_factor=.1, color=color),
#                 mlab.quiver3d(tvec[0],tvec[1],tvec[2],0,0,1, line_width=3, scale_factor=.1, color=color)]
#     num_pts = len(pts)
#     plane = None
#     # mesh_pts = []
#     # for i in range(3):
#     #     mesh_tmp = []
#     #     for j in range(num_pts):
#     #         mesh_tmp.append([pts[j,i],pts[(j+1)%num_pts,i]])
#     #     mesh_pts.append(mesh_tmp)
#     # plane = mlab.mesh(mesh_pts[0], mesh_pts[1], mesh_pts[2], color=color)
#     return vecs,plane

# def plot_epipolar(tvec, R=None, color=(1,0,0)):
#     pt1 = np.array([0.1,0,0], dtype=np.float32)
#     pt2 = np.array([0,0.1,0], dtype=np.float32)
#     pt3 = np.array([0,0,0.1], dtype=np.float32)

#     tvec = np.array(tvec).reshape((3))
#     vecs = []
#     pts = [pt1, pt2, pt3]
#     pts_rot = [np.matmul(R, p - np.array(tvec)) for p in pts]
#     ori = np.matmul(R, - np.array(tvec))
#     # print('pts_rot: \n', pts_rot)
#     # print('ori: \n', ori)
#     pts = np.array(pts_rot)
#     for i in range(3):
#         mlab.text3d(ori[0]+pts_rot[i][0]*0.1, ori[1]+pts_rot[i][1]*0.1, ori[2]+pts_rot[i][2]*0.1, str(i), scale=(0.01,0.01,0.01))
#         vec = mlab.quiver3d(ori[0],ori[1],ori[2],pts_rot[i][0], pts_rot[i][1], pts_rot[i][2], line_width=3, scale_factor=.1, color=color)
#         vecs.append(vec)

def visual_reprojection(img, pts3D, K, R, T):
    img = img_dots.copy()
    for p in pts3D:
        p = np.expand_dims(np.array(p), -1)
        p = np.matmul(K, np.matmul(R, p + T))
        p = (int(p[1,0]/p[2,0]), int(p[0,0]/p[2,0]))
        print('Plot point: ', p)
        cv2.circle(img, p, radius=10, thickness=-1, color=(200,0,0))
    return img

def get_RT(R, T):
    T = T.reshape(3,1)
    T_mat = np.concatenate((np.concatenate((np.identity(3), T), axis=1), np.array([0,0,0,1]).reshape(1,4)), axis=0)
    R_mat = np.concatenate((np.concatenate((R, np.zeros((3,1))), axis=1), np.array([0,0,0,1]).reshape(1,4)), axis=0)
    return np.matmul(T_mat, R_mat)

img_path = './data/CalibrationImgs/projector(test)/'
img_dots = cv2.imread("./data/test/dots.png")
find_unity_params = False
files = next(os.walk(img_path))[2]

count = 0
circleBoard = []
circleWorld = []
circleCam = []
projCirclePointsAccum = []
img_list = []
R_cam = 0
T_cam = 0
cam_to_board = 0
# figure = mlab.figure('visualize')
# cam_vec,cam_plane = plot_plane([0,0,0], color=(0,1,0))
for fname in files:
    if fname.endswith('.jpg'):
        frame = cv2.imread(img_path+fname)
        frame = cv2.resize(frame,(int(w_cam),int(h_cam)))
        img_list.append(frame.copy())
        print(fname)
        ret_charuco = get_charuco(frame.copy())

        if ret_charuco is not None:
            frame_charuco, charucoCorners, charucoIds, rvecs, tvecs = ret_charuco
            R, _ = cv2.Rodrigues(rvecs[0])
            R_cam = np.array(R)
            T_cam = np.array(tvecs[0]).reshape(3,1)
            cam_to_board = np.linalg.inv(get_RT(R_cam, T_cam))
            charucoCorners_world, origin = plot_charuco(R, tvecs[0], charucoCorners, charucoIds)
            # boad_vecs,board = plot_plane(origin, R, color=(1,0,0))

            # Find circle
            projCirclePoints = get_circle_coord(test_plot=False)
            circle_cam, ret_circle, ret_circle3d = get_circle(frame, [frame_charuco, charucoCorners, charucoIds, rvecs, origin], 0, charucoCorners, projCirclePoints=np.squeeze(projCirclePoints), init_param=True)
            circleCam.append(circle_cam)
            circleBoard.append(ret_circle)
            circleWorld.append(ret_circle3d)
            
            projCirclePointsAccum.append(projCirclePoints)

            # Visualize circle
            # print("circleBoard: \n", ret_circle.tolist())
            # print("circleWorld: \n", ret_circle3d.tolist())
            # circle_plot = [mlab.points3d(circle[0], circle[1], circle[2], scale_factor=0.01) for circle in ret_circle3d]
            # figure.scene.disable_render = True # Super duper trick
            # circle_text = [mlab.text3d(ret_circle3d[i,0], ret_circle3d[i,1], ret_circle3d[i,2], str(i), scale=(0.01,0.01,0.01)) for i in range(ret_circle3d.shape[0])]
            # for i in range(ret_circle3d.shape[0]):
            #     idx = mlab.text3d(ret_circle3d[i,0], ret_circle3d[i,1], ret_circle3d[i,2], str(i), scale=(0.01,0.01,0.01))
            # figure.scene.disable_render = False
            count += 1

circleBoard = np.array(circleBoard).astype('float32')
circleCam = np.array(circleCam).astype('float32')
circleWorld = np.array(circleWorld).astype('float32')
# print("circleBoard shape: ", circleBoard.shape, " circleCam shape: ", circleCam.shape, " circleWorld shape: ", circleWorld.shape)
projCirclePointsAccum = np.array(projCirclePointsAccum).astype('float32')
K_proj = np.array([[1500, 0, 1920/2.0],[0,1500,1080/2.0],[0,0,1]],dtype=np.float32)
dist_coef_proj = np.array([0.0, 0.0, 0.0, 0.0, 0.0],dtype=np.float32)
ret, K_proj, dist_coef_proj, rvecs, tvecs = cv2.calibrateCamera(circleBoard,
                                                                projCirclePointsAccum,
                                                                (1920, 1080),
                                                                K_proj,
                                                                dist_coef_proj,
                                                                #flags = cv2.CALIB_USE_INTRINSIC_GUESS)
                                                                flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K1  + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6)

print("proj calib mat after\n%s"%K_proj.tolist())
print("proj dist_coef %s"%dist_coef_proj.T)
print("calibration reproj err %s"%ret)
print("stereo calibration")
ret, K, dist_coef, K_proj, dist_coef_proj, R_cam_proj, T_cam_proj, _, _ = cv2.stereoCalibrate(
        circleWorld,
        circleCam,
        projCirclePointsAccum,
        K_cam.copy(),
        dist_coef,
        K_proj,
        dist_coef_proj,
        (1920,1080),
        flags = cv2.CALIB_FIX_INTRINSIC
        )
print('Reprojection error stereo: ', ret)
T_cam_proj = np.array(T_cam_proj)
R_cam_proj = np.array(R_cam_proj)
print("Translation from projector to camera: \n",(-1 * T_cam_proj).tolist())
print("Distance from projector to camera: ", np.linalg.norm(T_cam_proj))
#print('Rotation from projector to camera: \n', sp_rot.from_dcm(np.linalg.inv(R_cam_proj)).as_quat().tolist())
proj_to_cam = np.linalg.inv(get_RT(R_cam_proj, T_cam_proj))
proj_to_board = np.matmul(cam_to_board, proj_to_cam)
print("proj_to_cam: \n", proj_to_cam)
print("cam_to_board: \n", cam_to_board)
print("proj_to_board: \n", proj_to_board)
# print("Rotation from proj to board: \n", sp_rot.from_dcm(proj_to_board[0:3, 0:3]).as_quat().tolist())
# print("Position from proj to board: \n", np.matmul(proj_to_board, np.array([0,0,0,1]).reshape(4,1)).reshape(4,).tolist())
#proj_vecs, proj = plot_plane(T_cam_proj, R_cam_proj, color=(0,0,1))
#mayavi.mlab.show()
# Visualize reprojection
rvec_proj,_ = cv2.Rodrigues(R_cam_proj)
for circ3D in circleWorld:
    img_pts,_ = cv2.projectPoints(circ3D, rvec_proj, T_cam_proj, K_proj, dist_coef)
    img = img_dots.copy()
    for p in img_pts:
        cv2.circle(img, tuple(p[0,:]), radius=10, thickness=-1, color=(200,0,0))
    cv2.imshow('img', img)
    cv2.waitKey(0)


def from_to_rotation(a,b):
    v = np.cross(a,b)
    s = 1.0 * np.linalg.norm(v)
    c = np.dot(a,b)
    v_mat = np.array([[0, -1 * v[2], v[1]],[v[2], 0, -1 * v[0]],[-1 * v[1], v[0], 0]], dtype=np.float32)
    R = np.identity(3) + v_mat + (1-c)/s**2 * np.matmul(v_mat, v_mat)
    return R

def find_unity_transformations(pts3D, proj_to_board, cam_to_board, proj_to_cam, board_to_world, img_dots, K_proj=[[1049.9283171946713, 0.0, 995.6319450901574], [0.0, 1115.4836215220273, 1302.7626513835548], [0.0, 0.0, 1.0]], last_pt_to_world=[-0.174, 0.154, 0.], visualize=False, reproject=True):
    board_to_world_mat = np.concatenate((np.concatenate((np.identity(3), board_to_world.reshape(3,1)), axis=1), np.array([0,0,0,1]).reshape(1,4)), axis=0)
    board_to_world_4D = np.insert(board_to_world, 3, 1).reshape(4,1)
    # Convert the transformations to the world coordinate
    proj_to_world = np.matmul(board_to_world_mat, proj_to_board)
    cam_to_world = np.matmul(board_to_world_mat, cam_to_board)
    ptsPlane = [np.matmul(cam_to_board, np.insert(np.array(p), 3, 1).reshape(4,1)) + board_to_world_4D for p in pts3D]
    # Convert the points in board coordinate to the world coordinate
    R_board_world = from_to_rotation(cam_to_world[0:3,1], np.array([0,1,0])) # Align the camera's y-axis with the world's y-axis
    ptsPlane_corrected = [np.matmul(R_board_world, np.array([p[0,0], p[1,0], p[2,0]]).reshape(3,1)) for p in ptsPlane]
    last_pt = ptsPlane_corrected[-1]
    # Move the origin to the upper left corner of the table (manually measured)
    last_pt_corrected = np.array(last_pt_to_world).reshape(3,1)
    z_corrected = last_pt_corrected - last_pt
    ptsPlane_corrected = [p + z_corrected for p in ptsPlane_corrected]
    board_to_world_corrected = get_RT(R_board_world, z_corrected)
    proj_to_world = np.matmul(board_to_world_corrected, proj_to_world)
    cam_to_world = np.matmul(board_to_world_corrected, cam_to_world)
    # Convert from Cartesian to Unity
    T_cam = cam_to_world[0:3, 3].tolist()
    y_cam = (-1 * cam_to_world[0:3, 1]).tolist()
    z_cam = cam_to_world[0:3, 2].tolist()
    cam_param = [[T_cam[0],-1 * T_cam[1],T_cam[2]], [y_cam[0],-1 * y_cam[1],y_cam[2]], [z_cam[0],-1 * z_cam[1],z_cam[2]]]
    T_proj = proj_to_world[0:3, 3].tolist()
    y_proj = (-1 * proj_to_world[0:3, 1]).tolist()
    z_proj = proj_to_world[0:3, 2].tolist()
    proj_param = [[T_proj[0],-1 * T_proj[1],T_proj[2]], [y_proj[0],-1 * y_proj[1],y_proj[2]], [z_proj[0],-1 * z_proj[1],z_proj[2]]]
    print()
    print("Set camera y-axis:\n", cam_param[1])
    print("Set camera z-axis:\n", cam_param[2])
    print("Set camera translation:\n", cam_param[0])
    print()
    print("Set projector y-axis:\n", proj_param[1])
    print("Set projector z-axis:\n", proj_param[2])
    print("Set projector translation:\n", proj_param[0])

    # if visualize:
    #     figure = mlab.figure('visualize')
    #     circle_plot = [mlab.points3d(circle[0,0], circle[1,0], circle[2,0], scale_factor=0.01) for circle in ptsPlane_corrected]
    #     plot_plane(np.array([0,0,0]), np.identity(3), color=(1,0,0))
    #     proj_vecs, proj = plot_plane(proj_to_world[0:3,3], proj_to_world[0:3,0:3], color=(0,0,1))
    #     cam_vecs, cam = plot_plane(cam_to_world[0:3,3], cam_to_world[0:3,0:3], color=(0,1,0))
    #     mayavi.mlab.show()
    if reproject:
        world_to_proj = np.linalg.inv(proj_to_world)
        circ3D = [np.array([circle[0,0], circle[1,0], circle[2,0]], dtype=np.float32).reshape(3,1) for circle in ptsPlane_corrected]
        K_proj = np.array(K_proj)
        dist_coef = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        img_pts,_ = cv2.projectPoints(np.stack(circ3D), world_to_proj[0:3,0:3], world_to_proj[0:3,3], K_proj, dist_coef)
        img = img_dots.copy()
        for p in img_pts:
            cv2.circle(img, tuple(p[0,:]), radius=10, thickness=-1, color=(200,0,0))
        cv2.imshow('img', img)
        cv2.waitKey(0)
    return cam_param, proj_param

if find_unity_params:
    find_unity_transformations(np.squeeze(circleWorld).tolist(), proj_to_board, cam_to_board, proj_to_cam, board_to_world, img_dots, K_proj=K_proj, visualize=True, reproject=True)