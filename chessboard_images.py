import numpy as np
import cv2
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# prepare object points, like (0,0,0), (3,0,0), (6,0,0) ....,(21,12,0) because each square size is 3cm
objp = np.zeros((5*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:24:3, 0:15:3].T.reshape(-1, 2)


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = []

# Reading all images from the calibration data set
for filename in os.listdir('C:\\Users\ADMIN\Desktop\data_set'):
    img = cv2.imread(os.path.join('C:\\Users\ADMIN\Desktop\data_set', filename))
    if img is not None:
        images.append(img)


for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)

    # If not found, print a message
    if ret == False:
        print("CORNERS NOT FOUND")

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("CORNERS FOUND")
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (8, 5), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

    cv2.destroyAllWindows()


# Camera calibration and camera matrix calculation
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Printing Camera Parameters and distortion coefficients
np.set_printoptions(suppress=True)
print()
print("Camera Matrix : ")
print(mtx)
print()
print("Focal length along x - axis : ")
print(mtx[0][0])
print()
print("Focal length along y - axis : ")
print(mtx[1][1])
print()
print("Distortion Coefficients : ")
print(dist)
