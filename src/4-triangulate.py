import os
import numpy as np
import cv2

from scanner.triangulation import Triangulate
from scanner.utils import visualize




# Declare folders to use for triangulation
cam_result_folder = './data/calib_results/cam_1440/'
proj_result_folder = './data/calib_results/proj/'
stereo_result_folder = './data/calib_results/stereo_setups/groot/'
gray_code_folder = './data/recordings/groot/'
result_folder = './data/scan/test/'

# Load calibration data
cam_mtx = np.load(os.path.join(cam_result_folder,'cam_mtx.npy'))
cam_dist = np.load(os.path.join(cam_result_folder,'cam_dist.npy'))
proj_mtx = np.load(os.path.join(proj_result_folder,'proj_mtx.npy'))
proj_dist = np.load(os.path.join(proj_result_folder,'proj_dist.npy'))
proj_R = np.load(os.path.join(stereo_result_folder,'R.npy'))
proj_T = np.load(os.path.join(stereo_result_folder,'T.npy'))

# Load decoded gray codes
h_pixels = np.load(os.path.join(gray_code_folder,'h_pixels.npy'))
v_pixels = np.load(os.path.join(gray_code_folder,'v_pixels.npy'))

# Triangulate
triangulator = Triangulate(h_pixels, v_pixels, (2560, 1440), cam_mtx, cam_dist, (800, 600), proj_mtx, proj_dist, proj_R, proj_T, gray_code_folder)
cam_pts, proj_pts,  colors = triangulator.get_cam_proj_pts()
pts_3d = triangulator.triangulate(cam_pts, proj_pts)
pts_3d, colors = triangulator.filter_3d_pts(pts_3d, colors)

# Visualize point cloud
visualize.plot_point_cloud(pts_3d, colors)