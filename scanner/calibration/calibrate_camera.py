import numpy as np
import cv2 as cv
import glob
import os
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
dist = None
mtx = None
correctDist = True
calibration_folder = './CalibrationImgs'
if not os.path.exists(calibration_folder):
    os.mkdir(calibration_folder)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)    
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        # Draw and display the corners
        cv.drawChessboardCorners(gray, (7,6), corners2, ret)

    # Display the resulting frame
    if dist is not None and mtx is not None and correctDist:
        h,  w = frame.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv.undistort(gray, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imshow('frame', dst)
    else:
        cv.imshow('frame', gray)


    keyPressed = cv.waitKey(1)
    if keyPressed == ord('u'):
        correctDist = not correctDist
        if correctDist == True:
            print('Correct dist')
        else:
            print('Do not correct dist')
    if keyPressed == ord('q'):
        break
    if keyPressed == ord('s') and mtx is not None and dist is not None:
        np.save('./data/mtx.npy', mtx)
        np.save('./data/dist.npy', dist)
    if keyPressed == ord('l'):
        if os.path.exists('./data/mtx.npy'):
            mtx = np.load('./data/mtx.npy')
        if os.path.exists('./data/dist.npy'):
            dist = np.load('./data/dist.npy')
    if keyPressed == ord('c') and ret == True:
        images = glob.glob(os.path.join(calibration_folder,'*.jpg'))
        n = 0
        while os.path.exists(os.path.join(calibration_folder,f'calibrate_{n}.jpg')):
            n += 1
        cv.imwrite(os.path.join(calibration_folder,f'calibrate_{n}.jpg'), frame)
    if keyPressed == ord('k'):
        images = glob.glob(os.path.join(calibration_folder,'*.jpg'))
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (7,6), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
                objpoints.append(objp)

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print(mtx)
        print(dist)


cv.destroyAllWindows()