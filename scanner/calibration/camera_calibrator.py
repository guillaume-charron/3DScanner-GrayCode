import glob
import os
import cv2
import numpy as np

class CameraCalibrator(object):
    def __init__(self, 
                 cam_width=1920, 
                 cam_height=1080, 
                 cam_mtx=None, 
                 cam_dist=None):
        
        # Define the charuco board parameters
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.charuco_board = cv2.aruco.CharucoBoard((5, 7), 0.04, .02, self.dictionary)
        self.params = cv2.aruco.DetectorParameters()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

        # Camera parameters (image size and intrinsic parameters)
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_mtx = cam_mtx
        self.cam_dist = cam_dist

    def detect_markers(self, frame, gray, draw=True):
        """
        Detect markers and corners of the charuco board in the image

        Parameters:
        ---------
        frame : Mat
            image to draw detected markers
        gray : Mat 
            gray version of the image
        draw : bool, optional (default = True)
            draw detected markers or not
        
        Returns:
        -------
        ret : bool
            True if at least one marker and the board were detected
        frame : Mat
            Image with detected markers
        allCorners : list
            Corners of the detected markers
        allIds : list
            Ids of the detected markers
        """
        rvec = None
        tvec = None
        ret = False
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        # Detect markers and corners
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
        allCorners = []
        allIds = []
        if corners is not None and len(corners) > 0:
            if draw:
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for corner in corners:
                cv2.cornerSubPix(gray, corner, (3,3), (-1,-1), criteria)
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.charuco_board)
            if charuco_ids is not None and len(charuco_ids) > 0:
                if draw:
                    frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
                if self.cam_mtx is not None and self.cam_dist is not None:
                    ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.charuco_board, self.cam_mtx, self.cam_dist, None, None)

                    if ret and draw:
                        frame = cv2.drawFrameAxes(frame, self.cam_mtx, self.cam_dist, rvec, tvec, 0.1)

                allCorners = charuco_corners
                allIds = charuco_ids

        return ret, frame, allCorners, allIds

    def calibrate(self, calibration_folder, calib_flags=cv2.CALIB_USE_INTRINSIC_GUESS):
        """
        Calibrate the camera using the charuco board

        Parameters:
        ----------
        calibration_folder : str
            path to the folder containing the calibration images
        calib_flags : int, optional (default = cv2.CALIB_USE_INTRINSIC_GUESS)
            flags for the calibration function
        
        Returns:
        -------
        cam_mtx : Mat
            camera matrix
        cam_dist : Mat
            distortion coefficients
        """
        allCorners = []
        allIds = []

        # Calibrate camera
        images = glob.glob(os.path.join(calibration_folder,'*.jpg'))
        for fname in images:
            img = cv2.imread(fname)
            print('Reading image:', fname)

            # Convert to gray
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect markers and corners
            ret, _, corners, ids = self.detect_markers(img, gray, draw=False)

            if ret:
                allCorners.append(corners)
                allIds.append(ids)
            else:
                print('Bad image:', fname)

        print('Camera calibration')
        if self.cam_mtx is None or self.cam_dist is None:
            cameraMatrixInit = np.array([[ 1000.,    0., self.cam_width/2.],
                                         [    0., 1000., self.cam_height/2.],
                                         [    0.,    0.,           1.]])
            distCoeffsInit = np.zeros((5,1))
        else:
            cameraMatrixInit = self.cam_mtx
            distCoeffsInit = self.cam_dist

        ret, cam_mtx, cam_dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(allCorners, allIds, self.charuco_board, (self.cam_width,self.cam_height), cameraMatrixInit, distCoeffsInit, flags = calib_flags, criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
        print(cam_mtx)
        print(cam_dist)
        print('Error', ret)

        self.cam_mtx = cam_mtx
        self.cam_dist = cam_dist

        return cam_mtx, cam_dist