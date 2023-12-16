import glob
import os
import cv2
import numpy as np

class ProjCamCalibrator(object):
    def __init__(self, 
                 cam_width=1920, 
                 cam_height=1080, 
                 cam_mtx=None, 
                 cam_dist=None,
                 proj_width=1920,
                 proj_height=1080,
                 proj_mtx=None,
                 proj_dist=None,
                 circle_grid_size = (4, 11),
                 circle_r=15):
        
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

        # Projector parameters (image size and intrinsic parameters)
        self.proj_width = proj_width
        self.proj_height = proj_height
        self.proj_mtx = proj_mtx
        self.proj_dist = proj_dist

        # Generate circle grid
        self.circle_grid_size = circle_grid_size
        self.circle_r = circle_r
        self.circle_2d_pts = self.build_circle_grid_pts(circle_grid_size, circle_r)

    def detect_markers(self, frame, gray,draw=True):
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
        H : Mat
            Homography matrix of the camera to the charuco board
        """
        rvec = None
        tvec = None
        ret = False
        H = None
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        # Detect markers and corners
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)

        if corners is not None and len(corners) > 0:
            if draw:
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for corner in corners:
                cv2.cornerSubPix(gray, corner, (3,3), (-1,-1), criteria)

            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.charuco_board)
            if charuco_ids is not None and len(charuco_ids) > 0:
                if draw:
                    frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
                
                ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.charuco_board, self.cam_mtx, self.cam_dist, None, None)
                if ret and draw:
                        frame = cv2.drawFrameAxes(frame, self.cam_mtx, self.cam_dist, rvec, tvec, 0.1)

                obj_points = np.reshape(np.array(self.charuco_board.getObjPoints())[ids][:,:,:, :2], (len(corners)*4, 2))
                corners_flatten = np.reshape(corners, (len(corners)*4, 2))
                H, _ = cv2.findHomography(corners_flatten, obj_points, cv2.RANSAC, 5.0)

        return ret, frame, H

    def detect_circle_grid(self, frame, gray, H, draw=True):
        """
        Detect the circle grid in the image

        Parameters:
        ---------
        frame : Mat
            image to draw detected circles grid
        gray : Mat
            gray version of the image
        H : Mat
            Homography matrix of the camera to the charuco board
        draw : bool, optional (default = True)
            draw detected circles or not
        
        Returns:
        -------
        ret : bool
            True if the circle grid was detected
        frame : Mat
            Image with detected circles
        circles : Mat
            2D coordinates of the detected circles in the camera
        circles3D : Mat
            3D coordinates of the detected circles in the charuco board reference frame (z=0)
        """
        ret, circles = cv2.findCirclesGrid(gray, self.circle_grid_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)

        circles3D = None
        if ret:
            if draw:
                frame = cv2.drawChessboardCorners(frame, self.circle_grid_size, circles, ret)

            # Apply homography to get the 3D coordinates of the circles in the charuco board reference frame
            circles3D = cv2.perspectiveTransform(circles, H)
            # Add z=0 to the 3D coordinates
            circles3D = np.pad(circles3D, ((0,0), (0,0), (0,1)), 'constant', constant_values=0)
            circles3D = circles3D.astype(np.float32)

        return ret, frame, circles, circles3D

    def build_circle_grid_pts(self, grid_size, circle_r):
        """
        Build the 2D coordinates of the circle grid

        Parameters:
        ---------
        grid_size : tuple
            Number of circles in the grid (nb_col, nb_row)
        circle_r : int
            Radius of the circles
        
        Returns:
        -------
        circle_2d_pts : Mat
            2D coordinates of the circles in the grid
        """
        nb_col, nb_row = grid_size
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

    def get_circle_grid_image(self, x, y):
        """
        Return an image of the circle grid

        Parameters:
        ---------
        x : int
            x position of the grid
        y : int
            y position of the grid
        
        Returns:
        -------
        img_circle : Mat
            Image of the circle grid
        """
        img_circle = np.zeros((self.proj_height, self.proj_width, 4), dtype=np.uint8)
        self.circle_2d = self.circle_2d_pts + [x, y]
        white_padding = 3*self.circle_r
        min_x = np.min(self.circle_2d[:,0]) - white_padding
        max_x = np.max(self.circle_2d[:,0]) + white_padding
        min_y = np.min(self.circle_2d[:,1]) - white_padding
        max_y = np.max(self.circle_2d[:,1]) + white_padding
        img_circle[min_y:max_y, min_x:max_x] = cv2.bitwise_not(img_circle[min_y:max_y, min_x:max_x])
        for c in self.circle_2d:
            img_circle = cv2.circle(img_circle, tuple(c.astype(np.int32)), self.circle_r, (0,0,0), cv2.FILLED)
        return img_circle

    def calibrate(self, image_folder, calibrate_proj=True):
        """
        Calibrate the camera and the projector using the charuco board and the circle grid

        Parameters:
        ---------
        image_folder : str
            path to the folder containing the calibration images
        calibrate_proj : bool, optional (default = True)
            calibrate the projector or not
        
        Returns:
        -------
        proj_mtx : Mat
            projector intrinsic parameters matrix
        proj_dist : Mat
            projector distortion coefficients
        proj_R : Mat
            rotation matrix between the camera and the projector
        proj_T : Mat
            translation vector between the camera and the projector
        R1 : Mat
            rectification transform (rotation matrix) for the camera
        R2 : Mat
            rectification transform (rotation matrix) for the projector
        P1 : Mat
            projection matrix in the new (rectified) coordinate systems for the camera
        P2 : Mat
            projection matrix in the new (rectified) coordinate systems for the projector
        Q : Mat
            disparity-to-depth mapping matrix
        """
        proj_obj_pts = []
        proj_circle_pts = []
        cam_circle_pts = []
      
        # Calibrate projector-camera stereo setup
        images = glob.glob(os.path.join(image_folder,'*.jpg'))
        for fname in images:
            img = cv2.imread(fname)
            # Convert to gray
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect markers and corners
            ret, img, H = self.detect_markers(img, gray)

            if ret:
                # Find projected circles
                ret, img, circles_2d_cam, circles_3d = self.detect_circle_grid(img, gray, H)
                if ret:
                    proj_obj_pts.append(circles_3d)
                    proj_circle_pts.append(np.expand_dims(self.circle_2d.astype(np.float32), axis=1))
                    cam_circle_pts.append(circles_2d_cam)
                else:
                    print('Bad image:', fname)

        if calibrate_proj:
            if self.proj_mtx is None:
                self.proj_mtx = np.array([[ 3000.,    0., self.proj_width/2.],
                                          [    0., 3000., self.proj_height/2.],
                                          [    0.,    0.,           1.]])
            print('\nProjector calibration')
            ret, proj_mtx, proj_dist, _, _ = cv2.calibrateCamera(proj_obj_pts, proj_circle_pts, (self.proj_width, self.proj_height), self.proj_mtx, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
            print(proj_mtx)
            print(proj_dist)
            print('Error', ret)

            self.proj_mtx = proj_mtx
            self.proj_dist = proj_dist

        print('\nStereo calibration')
        error, cam_mtx, cam_dist, proj_mtx, proj_dist, proj_R, proj_T,_,_ = cv2.stereoCalibrate(proj_obj_pts, cam_circle_pts, proj_circle_pts, self.cam_mtx, self.cam_dist, self.proj_mtx, self.proj_dist, (2560,1440), flags=cv2.CALIB_FIX_INTRINSIC)
        print('Camera parameters')
        print(cam_mtx, cam_dist)
        print('Projector parameters')
        print(proj_mtx, proj_dist)
        print('Rotation and translation')
        print(proj_R)
        print(proj_T)
        print('Error', error)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cam_mtx, cam_dist, proj_mtx, proj_dist, (self.cam_width,self.cam_height), proj_R, proj_T)

        return proj_mtx, proj_dist, proj_R, proj_T, R1, R2, P1, P2, Q

