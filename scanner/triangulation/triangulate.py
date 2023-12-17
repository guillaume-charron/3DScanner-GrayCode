import numpy as np
import cv2

class Triangulate(object):
    def __init__(self, 
                 h_pixels=None, 
                 v_pixels=None,
                 cam_size=(1920, 1080), 
                 cam_mtx=None, 
                 cam_dist=None,
                 proj_size=(1920, 1080),
                 proj_calib_size=(1920, 1080), 
                 proj_mtx=None, 
                 proj_dist=None, 
                 proj_R=None, 
                 proj_T=None, 
                 image_folder=None):
        self.h_pixels = h_pixels
        self.v_pixels = v_pixels
        self.cam_w, self.cam_h = cam_size
        self.cam_mtx = cam_mtx
        self.cam_dist = cam_dist
        self.proj_w, self.proj_h = proj_size
        self.proj_mtx = proj_mtx
        self.proj_dist = proj_dist

        # Scale projection matrix
        if proj_mtx is not None:
            calib_w, calib_h = proj_calib_size
            scale_x = self.proj_w / calib_w
            scale_y = self.proj_h / calib_h
            self.proj_mtx[0, :] = self.proj_mtx[0, :] * scale_x
            self.proj_mtx[1, :] = self.proj_mtx[1, :] * scale_y

        self.proj_T = proj_T
        self.proj_R = proj_R
        self.image_folder = image_folder
    
    def get_cam_proj_pts(self, img_white=None):
        """
        Get 2D points from camera and projector with their color

        Returns:
            cam_pts: 2D points from camera
            proj_pts: 2D points from projector
            colors: colors of the points
        """
        cam_pts = []
        proj_pts = []
        colors = []

        for i in range(self.cam_w):
            for j in range(self.cam_h):
                h_value = self.h_pixels[j, i]
                v_value = self.v_pixels[j, i]
                if h_value == -1 or v_value == -1:
                    pass
                else:
                    cam_pts.append([i,j])
                    h_value = min(self.proj_w-1, h_value)
                    v_value = min(self.proj_h-1, v_value)
                    proj_pts.append([ h_value, v_value])
                    if img_white is not None:
                        colors.append(img_white[j, i, :])

        cam_pts = np.array(cam_pts, dtype=np.float32)
        proj_pts = np.array(proj_pts, dtype=np.float32)
        if img_white is not None:
            colors_array = np.array(colors).astype(np.float64)/255.0

        return cam_pts, proj_pts, colors_array
    
    def triangulate(self, cam_pts, proj_pts):
        """
        Triangulate 3D points from 2D points from camera and projector

        Parameters:
            cam_pts: 2D points from camera
            proj_pts: 2D points from projector

        Returns:
            points: 3D points
        """
        cam_pts_homo = cv2.convertPointsToHomogeneous(cv2.undistortPoints(np.expand_dims(cam_pts, axis=1), self.cam_mtx, self.cam_dist, R=self.proj_R))[:,0].T
        proj_pts_homo = cv2.convertPointsToHomogeneous(cv2.undistortPoints(np.expand_dims(proj_pts, axis=1), self.proj_mtx, self.proj_dist))[:,0].T
        T = self.proj_T[:,0]

        # Took from https://github.com/caoandong/Projector_Calibration/blob/master/visual_calib.py
        TLen = np.linalg.norm(T)
        NormedL = cam_pts_homo/np.linalg.norm(cam_pts_homo, axis=0)
        alpha = np.arccos(np.dot(-T, NormedL)/TLen)
        beta = np.arccos(np.dot(T, proj_pts_homo)/(TLen*np.linalg.norm(proj_pts_homo, axis=0)))
        gamma = np.pi - alpha - beta
        P_len = TLen*np.sin(beta)/np.sin(gamma)
        Pts = NormedL*P_len

        return Pts
    
    def filter_3d_pts(self, Pts, colors, threshold=.5):
        """
        Filter 3d point cloud to have a cleaner version

        Parameters
        ----------
        Pts: Mat
            Point cloud to filter
        colors : Mat
            Colors associated to each point
        threshold : float (optional) default=.5
            Treshold to use for filtering coordinates
        
        Returns
        ---------
        Pts_filtered: Mat
            The filtered point cloud
        colors_filtered : Mat
            The filtered colors
        """
        filter = (Pts[2] < threshold) & (Pts[2] > -threshold) & (Pts[1] < threshold) & (Pts[1] > -threshold) & (Pts[0] < threshold) & (Pts[0] > -threshold)
        Pts_filtered = Pts[:, filter]
        colors_filtered = colors[filter]
        return Pts_filtered, colors_filtered
