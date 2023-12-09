from time import sleep
import cv2
from scanner.acquisition import Camera
from scanner.grayCode.generate_codes import get_gray_codes, get_image_sequence

if __name__ == '__main__':
    cam = Camera()

    # Generate Gray code image sequence
    width, height = (800, 600)
    gray_codes = get_gray_codes(width, height)
    image_seq = get_image_sequence(gray_codes, width, height)
    seq_id = 0

    cv2.namedWindow('Gray Code Pattern', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Gray Code Pattern', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        cv2.imshow("Gray Code Pattern", image_seq[seq_id])
        sleep(0.5)

        frame = cam.get_frame()

        if frame is not None:
            frame = cv2.resize(frame, (960, 540))
            cv2.imshow('Camera', frame)

            if seq_id == len(image_seq) - 1: # Completed a sequence, analyze gray codes
                pass
            
            seq_id = (seq_id + 1) % len(image_seq)

        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            cam.stop_cam()
            break