

import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import PyGnuplot as pg
from mpl_toolkits.mplot3d import Axes3D

x1 = []
y1 = []
z1 = []
r22 = []
T_r = []

pg.default_term = 'x11'

np.set_printoptions(suppress=True)
print(cv2.__version__)

# Read video file
vidcap = cv2.VideoCapture("/Users/Akanksha/Desktop/cs9645-assign-4.mov")


# Find image frames of video
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        cv2.imwrite("frame_"+str(int(sec))+"newnew.jpg", image)     # save frame as JPG file
    return hasFrames


sec = 0
frameRate = 1
success = getFrame(sec)
while success:
    sec = sec + frameRate
    sec = round(sec, 1)
    success = getFrame(sec)

print("No. of frames = ", sec)
print()
print()

# Find point correspondences for two successive frames in video
# With these correspondences, estimate the fundamental matrix using RANSAC
MIN_MATCH_COUNT = 8
for i in range(0, 8):
    print("PARAMETERS FOR FRAMES "+str(i)+" and " + str(i+1))
    print()
    img1 = cv2.imread("frame_"+str(i)+"newnew.jpg", 0)
    img2 = cv2.imread("frame_new"+str(i+1)+"newnew.jpg", 0)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img1, None)
    des1 = np.float32(des1)
    des2 = np.float32(des2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

        src_pts = np.array(src_pts, dtype=np.int32)
        dst_pts = np.array(dst_pts, dtype=np.int32)

        fundamental_mat, mask = cv2.findFundamentalMat(points1=src_pts,
                                                   points2=dst_pts,
                                                   method=cv2.FM_8POINT + cv2.FM_RANSAC,
                                                   ransacReprojThreshold=0.01,
                                                   confidence=0.9999999)

        img_points1 = src_pts[mask.ravel() == 1]
        img_points2 = dst_pts[mask.ravel() == 1]
        print("Fundamental Matrix F:\n", fundamental_mat)

    camera_matrix = np.float32([[1111.341157, 0, 649.57225398], [0, 831.45917801, 358.44247886], [0, 0, 1]])

    # Estimate essential matrix
    E = camera_matrix.astype(np.float64).T.dot(fundamental_mat.astype(np.float64)).dot(camera_matrix.astype(np.float64))
    print()
    print("Essential Matrix E:\n", E)

    # Decompose essential matrix into R and T
    R, r2, T = cv2.decomposeEssentialMat(E)
    r22.append(R)
    T_r.append(T)
    print()
    print("Rotation Matrix R:\n", R)
    print()
    print("Translational Matrix T:\n", T)
    print()

    # Estimate camera poses using C = - (R^t).T
    r22[i] = np.transpose(r22[i])

    if i >= 1:
        r22[i] = np.matmul(r22[i], r22[i - 1])
        T_r[i] = np.matrix(T_r[i])
        T_r[i - 1] = np.matrix(T_r[i - 1])
        T_r[i] = T_r[i] + T_r[i - 1]

    C = -np.matmul(r22[i], T_r[i])
    print("Camera Pose")
    print(C)

# iterate over all point correspondences used in the estimation of the
# fundamental matrix
    camera_matrix_inv = np.linalg.inv(camera_matrix)
    first_inliers = []
    second_inliers = []
    for i in range(len(mask)):
        if mask[i]:
            # normalize and homogenize the image coordinates
            first_inliers.append(camera_matrix_inv.dot([src_pts[i][0],
                            dst_pts[i][1], 1.0]))
            second_inliers.append(camera_matrix_inv.dot([dst_pts[i][0],
                            src_pts[i][1], 1.0]))

    Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    Rt2 = np.hstack((r2, T.reshape(3, 1)))

    first_inliers = np.array(first_inliers).reshape(-1, 3)[:, :2]
    second_inliers = np.array(second_inliers).reshape(-1, 3)[:, :2]
    pts4D = cv2.triangulatePoints(Rt1, Rt2, first_inliers.T, second_inliers.T).T

# convert from homogeneous coordinates to 3D
    pts3D = pts4D[:, :3]/np.repeat(pts4D[:, 3], 3).reshape(-1, 3)

# plot with matplotlib
    Ys = pts3D[:, 0]
    Zs = pts3D[:, 1]
    Xs = pts3D[:, 2]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(Xs, Ys, Zs, c='r', marker='o')
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_zlabel('X')
    plt.title('3D point cloud: Use pan axes button below to inspect')
    plt.show()

    x1.append(C[0])
    y1.append(C[1])
    z1.append(C[2])

    print()
    print("----------------------------------------------------")
    print()

# Save values of camera pose in a file
# pg.s([x1, y1, z1], filename='assg4_9frames.txt')

# Plot camera trajectory
pg.c('splot "assg4_9frames.txt" using 1:2:3 with lines ')

