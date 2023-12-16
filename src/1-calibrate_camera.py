import os
import cv2
import numpy as np

from scanner.acquisition import Camera
from scanner.calibration import CameraCalibrator

# Declare folders to use for calibration
result_folder = './data/calib_results/cam_1440/'
calibration_folder = './data/CalibrationImgs/camera_1440/'

# Create folders if they don't exist
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
if not os.path.exists(calibration_folder):
    os.makedirs(calibration_folder)

# Get previous calibration data for the camera
cam_mtx = None
cam_dist = None
if os.path.exists(os.path.join(result_folder,'cam_mtx.npy')):
    cam_mtx = np.load(os.path.join(result_folder,'cam_mtx.npy'))
if os.path.exists(os.path.join(result_folder,'cam_dist.npy')):
    cam_dist = np.load(os.path.join(result_folder,'cam_dist.npy'))

# Camera parameters
cam_width, cam_height = (2560, 1440)
cam_src = 0
cam_fps = 20

# Remove distortion from camera frame
remove_dist = False

if __name__ == '__main__':
    # Start camera and calibrator
    cam = Camera(cam_src, width=cam_width, height=cam_height, fps=cam_fps)
    calibrator = CameraCalibrator(cam_width, cam_height, cam_mtx, cam_dist)

    while True:
        frame = cam.get_frame()

        if frame is not None:
            # Reset parameters
            og_frame = frame.copy()

            # Convert to gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers and corners
            ret, frame, _, _ = calibrator.detect_markers(frame, gray)

            if remove_dist:
                # Undistort image
                frame = cam.remove_dist(frame)
            
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
        if keyPressed == ord('k'):
            cam_mtx, cam_dist = calibrator.calibrate(calibration_folder)
            cam.mtx = cam_mtx
            cam.dist = cam_dist

            # Save calibration results
            np.save(os.path.join(result_folder, 'cam_mtx.npy'), cam_mtx)
            np.save(os.path.join(result_folder, 'cam_dist.npy'), cam_dist)