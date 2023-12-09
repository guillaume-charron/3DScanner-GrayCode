import cv2
import os
video_name = './data/recordings/tete_2'
vidcap = cv2.VideoCapture(f'{video_name}.mp4')
success,image = vidcap.read()
count = 0
if not os.path.exists(video_name):
    os.mkdir(video_name)
while success:
  cv2.imwrite(f"{video_name}/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  count += 1