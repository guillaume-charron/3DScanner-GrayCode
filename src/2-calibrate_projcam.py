import os
import cv2
import numpy as np

from scanner.acquisition import Camera
from scanner.calibration import ProjCamCalibrator

# Declare folders to use for calibration
cam_result_folder = './data/calib_results/cam_1440/'
proj_result_folder = './data/calib_results/proj/'
result_folder = './data/calib_results/stereo_setups/testStereo/'
image_folder = './data/CalibrationImgs/projector/'

# Create folders if they don't exist
if not os.path.exists(cam_result_folder):
    os.makedirs(cam_result_folder)
if not os.path.exists(proj_result_folder):
    os.makedirs(proj_result_folder)
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Get previous calibration data for the camera
cam_mtx = None
cam_dist = None
if os.path.exists(os.path.join(cam_result_folder,'cam_mtx.npy')):
    cam_mtx = np.load(os.path.join(cam_result_folder,'cam_mtx.npy'))
if os.path.exists(os.path.join(cam_result_folder,'cam_dist.npy')):
    cam_dist = np.load(os.path.join(cam_result_folder,'cam_dist.npy'))

# Get previous calibration data for the projector
proj_mtx = None
proj_dist = None
if os.path.exists(os.path.join(proj_result_folder,'proj_mtx.npy')):
    cam_mtx = np.load(os.path.join(proj_result_folder,'proj_mtx.npy'))
if os.path.exists(os.path.join(proj_result_folder,'proj_dist.npy')):
    cam_dist = np.load(os.path.join(proj_result_folder,'proj_dist.npy'))

# Camera parameters
cam_width, cam_height = (2560, 1440)
cam_src = 0
cam_fps = 20

# Projector parameters
proj_width, proj_height = (1920, 1080)
calibrate_proj = True

# Circle grid parameters
circle_grid_size = (4, 11)
circle_r = 15
default_x, default_y = (800, 350)
img_circle = np.zeros((1080,1920,4), dtype=np.uint8) # Empty image to draw circles

if __name__ == '__main__':
    # Start camera and calibrator
    cam = Camera(cam_src, width=cam_width, height=cam_height, fps=cam_fps)
    calibrator = ProjCamCalibrator(cam_width, cam_height, cam_mtx, cam_dist, proj_width, proj_height, proj_mtx, proj_dist, circle_grid_size, circle_r)
    
    # Generate circle grid image
    proj_img = calibrator.get_circle_grid_image(default_x, default_y)

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
            ret, frame, H = calibrator.detect_markers(frame, gray)

            if ret:
                # Find projected circles
                ret, frame, circles_2d_cam, circles_3d = calibrator.detect_circle_grid(frame, gray, H)

            if ret:
                consecutive_frames += 1
            else:
                consecutive_frames = 0

            if consecutive_frames > 5:
                n = 0
                while os.path.exists(os.path.join(image_folder,f'calibrate_{n}.jpg')):
                    n += 1
                im_name = os.path.join(image_folder,f'calibrate_{n}.jpg')
                cv2.imwrite(im_name, og_frame)
                print(f'Saved calibration image to {im_name}')
                consecutive_frames = 0

            # Show camera frame image
            resized_frame = cv2.resize(frame, (960, 540))
            cv2.imshow('frame', resized_frame)

            # Project circles grid
            cv2.imshow('circle', proj_img)

        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            cam.stop_cam()
            break
        if keyPressed == ord('c'):
            if circles_3d is not None:
                n = 0
                while os.path.exists(os.path.join(image_folder,f'calibrate_{n}.jpg')):
                    n += 1
                cv2.imwrite(os.path.join(image_folder,f'calibrate_{n}.jpg'), og_frame)
        if keyPressed == ord('k'):
            # Calibrate projector and stereo configuration
            try:
                proj_mtx, proj_dist, proj_R, proj_T, R1, R2, P1, P2, Q = calibrator.calibrate(image_folder, result_folder, calibrate_proj=calibrate_proj)

                if calibrate_proj:
                    # Save calibration results
                    np.save(os.path.join(proj_result_folder, 'proj_mtx.npy'), proj_mtx)
                    np.save(os.path.join(proj_result_folder, 'proj_dist.npy'), proj_dist)

                # Save stereo calibration results
                np.save(os.path.join(result_folder, 'proj_R.npy'), proj_R)
                np.save(os.path.join(result_folder, 'proj_T.npy'), proj_T)
                np.save(os.path.join(result_folder, 'R1.npy'), R1)
                np.save(os.path.join(result_folder, 'R2.npy'), R2)
                np.save(os.path.join(result_folder, 'P1.npy'), P1)
                np.save(os.path.join(result_folder, 'P2.npy'), P2)
            except Exception as e:
                print('Calibration failed!', e)