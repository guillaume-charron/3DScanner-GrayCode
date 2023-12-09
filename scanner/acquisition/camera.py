import os
from threading import Thread
import time
import cv2
import numpy as np

class Camera(object):
    def __init__(self, src=0, width=1920, height=1080, fps=30):
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
       
        # Set FPS wait time
        self.fps = fps
        self.SECONDS_PER_IMAGE = 1/fps
        self.WAIT_MS = int(self.SECONDS_PER_IMAGE * 1000)
        
        # Start frame retrieval thread
        self.thread_active = True
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        # Get Intrinsics parameters
        self.mtx = None
        self.dist = None
        if os.path.exists('./data/mtx.npy'):
            self.mtx = np.load('./data/mtx.npy')
        if os.path.exists('./data/dist.npy'):
            self.dist = np.load('./data/dist.npy')

        self.isNewFrame = False

    def update(self):
        while self.thread_active:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                self.isNewFrame = True
            time.sleep(self.SECONDS_PER_IMAGE)
    
    def get_frame(self):
        if self.isNewFrame:
            self.isNewFrame = False
            return self.frame
        return None

    def show_frame(self):
        #im = self.remove_dist()
        cv2.imshow('frame', self.frame)
        #cv2.waitKey(self.WAIT_MS)

    def remove_dist(self):
        h,  w = self.frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
        dst = cv2.undistort(self.frame, self.mtx, self.dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        if self.out_width != dst.shape[1]:
            self.out_width = dst.shape[1]
            self.out_height = dst.shape[0]
        return dst
    
    def stop_cam(self):
        self.thread_active = False
        self.capture.release()
        cv2.destroyAllWindows()