import os
import numpy as np
import cv2
import open3d as o3d
from open3d import core as o3c

class Triangulate(object):
    def __init__(self, 
                 h_pixels, 
                 v_pixels,
                 cam_size, 
                 cam_mtx, 
                 cam_dist,
                 proj_size, 
                 proj_mtx, 
                 proj_dist, 
                 proj_R, 
                 proj_T, 
                 image_folder):
        self.h_pixels = h_pixels
        self.v_pixels = v_pixels
        self.cam_w, self.cam_h = cam_size
        self.cam_mtx = cam_mtx
        self.cam_dist = cam_dist
        self.proj_w, self.proj_h = proj_size
        self.proj_mtx = proj_mtx
        self.proj_dist = proj_dist
        self.proj_T = proj_T
        self.proj_R = proj_R
        self.image_folder = image_folder
    
    def get_cam_proj_pts(self):
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

        # Get white image to extract colors
        img_white = cv2.imread(os.path.join(self.image_folder, 'frame_1.jpg'), cv2.IMREAD_COLOR)
        img_white = cv2.cvtColor(img_white, cv2.COLOR_BGR2RGB)

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
                    colors.append(img_white[j, i, :])

        cam_pts = np.array(cam_pts, dtype=np.float32)
        proj_pts = np.array(proj_pts, dtype=np.float32)
        colors_array = np.array(colors).astype(np.float64)/255.0

        return cam_pts, proj_pts, colors_array
    
    def triangulate(self, cam_pts, proj_pts):
        """
        Triangulate 3D points from 2D points from camera and projector

        Args:
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
        degalpha = alpha*180/np.pi
        beta = np.arccos(np.dot(T, proj_pts_homo)/(TLen*np.linalg.norm(proj_pts_homo, axis=0)))
        degbeta = beta*180/np.pi
        gamma = np.pi - alpha - beta
        P_len = TLen*np.sin(beta)/np.sin(gamma)
        Pts = NormedL*P_len

        return Pts
    
    def filter_3d_pts(self, Pts, colors):
        filter = (Pts[2] < 1.5) & (Pts[2] > -0.5)
        Pts_filtered = Pts[:, filter]
        colors_filtered = colors[filter]
        return Pts_filtered, colors_filtered
