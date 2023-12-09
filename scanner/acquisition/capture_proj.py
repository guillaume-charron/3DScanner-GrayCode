import os
import time
import cv2
import numpy as np
from scanner.acquisition import Camera
from scanner.grayCode.generate_codes import get_gray_codes, get_image_sequence
from scanner.grayCode.decode_codes import decode_images

if __name__ == '__main__':
    cam = Camera()

    # Parameters
    wait_time = 0.2 # in seconds

    # Generate Gray code image sequence
    width, height = (800, 600)
    gray_codes = get_gray_codes(width, height)
    image_seq = get_image_sequence(gray_codes, width, height)
    seq_len = len(image_seq)
    seq_id = 0
    gray_images = None

    # Start a full screen window for gray codes
    cv2.namedWindow('Gray Code Pattern', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Gray Code Pattern', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Get output data path
    recording_id = 0
    output_path = './data/recordings/record_{}'
    while os.path.exists(output_path.format(recording_id)):
        recording_id += 1
    output_path = output_path.format(recording_id)
    os.mkdir(output_path)

    image_id = 0
    is_ready = True
    start_time = -1.0

    while True:

        if is_ready:
            is_ready = False
            cv2.imshow("Gray Code Pattern", image_seq[seq_id])
            start_time = time.time()

        frame = cam.get_frame()

        if frame is not None and time.time() - start_time > (wait_time if image_id > 0 else 3*wait_time):
            frame_resized = cv2.resize(frame, (960, 540))
            cv2.imshow('Camera', frame_resized)
            cv2.imwrite(os.path.join(output_path, 'frame_{}.png'.format(image_id)), frame)
            image_id += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if gray_images is None:
                gray_images = np.zeros((seq_len, gray.shape[0], gray.shape[1]))
            gray_images[seq_id] = gray

            if seq_id == seq_len - 1: # Completed a sequence, analyze gray codes
                # Decode gray codes
                h_pixel, v_pixel = decode_images(gray_images)

                # TODO : Save data

                # TODO : Average with previous

                # TODO : Compute 3D points?
            
            seq_id = (seq_id + 1) % len(image_seq)
            is_ready = True

        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            cam.stop_cam()
            break