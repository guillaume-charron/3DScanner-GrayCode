import cv2
import numpy as np
from itertools import chain

from acquisition.threaded_film_pattern import Camera

# Define the charuco board parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
charuco_board = cv2.aruco.CharucoBoard((5, 7), 0.04, .02, dictionary)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
charucoParams = cv2.aruco.CharucoParameters()
refineParams = cv2.aruco.RefineParameters()
detector = cv2.aruco.CharucoDetector(charuco_board, charucoParams, params, refineParams)


#cv2.imwrite('charuco.jpg', charuco_board.generateImage((200*3, 200*3), 10, 1))

if __name__ == '__main__':
    cam = Camera()
    lastFrame = None
    while True:
        #cam.show_frame()
        frame = cam.get_frame()
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect markers and corners
            corners, ids, charuco_corners , charuco_ids = detector.detectBoard(gray)
            #_, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
            charuco_corners = np.array(list(chain.from_iterable(charuco_corners))).reshape(-1, 4, 2)
            if charuco_corners is not None and charuco_ids is not None:
                #print(charuco_corners.shape, charuco_ids.shape)
                cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners)
            cv2.imshow('frame', frame)



        #     lastFrame = frame
        # if lastFrame is not None:
        #     cv2.imshow('frame', lastFrame)	


        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            break


# # Create arrays to store object points and image points from all calibration images
# obj_points = []  # 3D points in real world space
# img_points = []  # 2D points in image plane

# # Load calibration images
# calibration_images = glob.glob('calibration_images/*.jpg')

# # Iterate through calibration images
# for image_path in calibration_images:
#     # Load image
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Detect markers and corners
#     corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
#     _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)

#     # If charuco corners are detected, add them to the calibration data
#     if charuco_corners is not None and charuco_ids is not None:
#         obj_points.append(charuco_board.chessboardCorners)
#         img_points.append(charuco_corners)

# # Calibrate the camera and projector
# ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(obj_points, img_points, charuco_board, gray.shape[::-1], None, None)

# # Print the calibration results
# print("Camera matrix:")
# print(camera_matrix)
# print("Distortion coefficients:")
# print(dist_coeffs)
# print("Rotation vectors:")
# print(rvecs)
# print("Translation vectors:")
# print(tvecs)
