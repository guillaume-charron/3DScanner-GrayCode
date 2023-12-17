import os
import cv2
import numpy as np

from scanner.triangulation import Triangulate
from scanner.utils import visualize

# Declare folders to use for triangulation
cam_result_folder = './data/calib_results/cam_1440/'
proj_result_folder = './data/calib_results/proj/'
stereo_result_folder = './data/calib_results/stereo_setups/2023-12-16/'
gray_code_folder = './data/recordings/groot_720p/'
output_folder = './data/point_clouds/groot_720/'

# Create folders if they don't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if os.path.exists(os.path.join(output_folder, 'pts_3d.npy')):
    print('Point cloud already exists. Loading point cloud...')
    pts_3d = np.load(os.path.join(output_folder, 'pts_3d.npy'))
    colors = np.load(os.path.join(output_folder, 'colors.npy'))

    triangulator = Triangulate()

else:
    print('Point cloud does not exist. Triangulating...')
    # Get white image to extract colors
    img_white = cv2.imread(os.path.join(gray_code_folder, 'frame_1.jpg'), cv2.IMREAD_COLOR)
    img_white = cv2.cvtColor(img_white, cv2.COLOR_BGR2RGB)

    # Set resolutions
    proj_w, proj_h = (1280, 720)
    calib_proj_w, calib_proj_h = (1920, 1080)
    cam_w, cam_h = (img_white.shape[1], img_white.shape[0])

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
    triangulator = Triangulate(h_pixels, 
                            v_pixels, 
                            (cam_w, cam_h), 
                            cam_mtx, 
                            cam_dist, 
                            (proj_w, proj_h),
                            (calib_proj_w, calib_proj_h),
                            proj_mtx, 
                            proj_dist, 
                            proj_R, 
                            proj_T, 
                            gray_code_folder)

    cam_pts, proj_pts, colors = triangulator.get_cam_proj_pts(img_white)
    pts_3d = triangulator.triangulate(cam_pts, proj_pts)

    # Save point cloud
    np.save(os.path.join(output_folder,'pts_3d.npy'), pts_3d)
    np.save(os.path.join(output_folder,'colors.npy'), colors)

# Filter point cloud
pts_3d, colors = triangulator.filter_3d_pts(pts_3d, colors, threshold=0.5)

# Visualize point cloud
visualize.plot_point_cloud(pts_3d, colors)