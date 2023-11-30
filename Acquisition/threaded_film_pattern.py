import os
from threading import Thread
import time
import cv2
import numpy as np

class Camera(object):
    def __init__(self, src=0, width=1920, height=1080, fps=30, out_file_name='./data/recordings/record_{}.mp4'):
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
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.out = None
        self.video_id = 0
        self.isRecording = False
        self.file_name = out_file_name
        self.mtx = None
        self.dist = None
        if os.path.exists('./data/mtx.npy'):
            self.mtx = np.load('./data/mtx.npy')
        if os.path.exists('./data/dist.npy'):
            self.dist = np.load('./data/dist.npy')
        self.out_width  = width
        self.out_height = height

        
    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.SECONDS_PER_IMAGE)
            
    def show_frame(self):
        im = self.remove_dist()
        if self.isRecording:
            self.out.write(im)
        cv2.imshow('frame', im)
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
    
    def start_recording(self):
        if not self.isRecording:
            self.isRecording = True
            self.video_id = 0
            while os.path.exists(self.file_name.format(self.video_id)):
                self.video_id += 1
            save_to = self.file_name.format(self.video_id)
            print('Start recording, save to:',save_to)
            self.out = cv2.VideoWriter(save_to,cv2.VideoWriter_fourcc(*'mp4v'), float(self.fps), (self.out_width, self.out_height),True)

    def stop_recording(self):
        if self.isRecording:
            self.isRecording = False
            self.out.release()
            print('Stop recording, saved to:', self.file_name.format(self.video_id))

if __name__ == '__main__':
    cam = Camera()
    while True:
        try:
            cam.show_frame()
        except AttributeError:
            pass

        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            break
        if keyPressed == ord('r'):
            cam.start_recording()
        if keyPressed == ord('s'):
            cam.stop_recording()



