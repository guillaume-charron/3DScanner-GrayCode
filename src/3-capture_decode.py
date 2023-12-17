import os
import time
import cv2
import numpy as np
from scanner.acquisition import Camera
from scanner.grayCode.generate_codes import get_gray_codes, get_image_sequence
from scanner.grayCode.decode_codes import get_codes, gray_to_decimal
from scanner.utils import visualize

# Camera parameters
cam_width, cam_height = (1920, 1080)
cam_src = 0
cam_fps = 30

# Projector parameters
proj_width, proj_height = (1920, 1080)

if __name__ == '__main__':
    cam = Camera(cam_src, width=cam_width, height=cam_height, fps=cam_fps)

    # Parameters
    wait_time = 0.3 # in seconds

    # Generate Gray code image sequence
    gray_codes = get_gray_codes(proj_width, proj_height)
    image_seq = get_image_sequence(gray_codes, proj_width, proj_height)
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
    cache_h_codes = []
    cache_v_codes = []
    MAX_NB_RUNS = 2
    run_id = 0

    while run_id < MAX_NB_RUNS:

        if is_ready:
            is_ready = False
            cv2.imshow("Gray Code Pattern", image_seq[seq_id])
            start_time = time.time()

        frame = cam.get_frame()

        if frame is not None and time.time() - start_time > (wait_time if image_id > 0 else 3*wait_time):
            frame_resized = cv2.resize(frame, (960, 540))
            cv2.imshow('Camera', frame_resized)
            cv2.imwrite(os.path.join(output_path, 'frame_{}.jpg'.format(image_id)), frame)
            image_id += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if gray_images is None:
                gray_images = np.zeros((seq_len, gray.shape[0], gray.shape[1]))
            gray_images[seq_id] = gray

            if seq_id == seq_len - 1: # Completed a sequence, analyze gray codes

                # Decode gray codes
                h_codes, v_codes = get_codes(gray_images)

                # Save data to cache
                cache_h_codes.append(h_codes)
                cache_v_codes.append(v_codes)

                run_id += 1
            
            seq_id = (seq_id + 1) % len(image_seq)
            is_ready = True

        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            cam.stop_cam()
            exit(0)

    # Stop camera
    cam.stop_cam()

    # Get maximum with previous (eliminate undefined pixels)
    best_h_codes = np.max(cache_h_codes, axis=0)
    best_v_codes = np.max(cache_v_codes, axis=0)

    # Decode gray codes
    h_pixels = np.array([gray_to_decimal(best_h_codes[:, y, x])  for y in range(0, best_h_codes.shape[1]) for x in range(0, best_h_codes.shape[2])]).reshape((best_h_codes.shape[1], best_h_codes.shape[2]))
    v_pixels = np.array([gray_to_decimal(np.flip(best_v_codes[:, y, x]))  for y in range(0, best_v_codes.shape[1])for x in range(0, best_v_codes.shape[2])] ).reshape((best_v_codes.shape[1], best_v_codes.shape[2]))

    # Save data to files
    np.save(os.path.join(output_path, 'h_pixels.npy'), h_pixels)
    np.save(os.path.join(output_path, 'v_pixels.npy'), v_pixels)

    # Display decoded gray codes
    color_map_img = visualize.create_color_map(proj_width, proj_height)
    visualize.plot_decoded_graycodes(h_pixels, v_pixels, color_map_img)
