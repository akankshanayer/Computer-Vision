import numpy as np
import cv2
import math
import random
import statistics
from numpy import matrix
import matplotlib.pyplot as plt

diff_arr = []
# 3D points in reference frame of stereoscopic vision system
P = np.float64([[0, 3, 50], [2, -5, 47], [-1, 7, 60], [5, -1, 40], [0, 2, 45], [3, -4, 44]])
# Gaze vectors in reference frame of eye tracker
g = np.float64([[0, 3, 30], [-3, -5, 28], [10, 7, 31], [-10, 1, 25], [-5, 2, 30], [-6, -4, 27]])
M_c = np.float64([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

p, _ = cv2.projectPoints(g, np.zeros(3), np.zeros(3), M_c, np.zeros(5))
dist_coef = np.zeros(4)

# Giving local rvec and tvec as input to solvePnP function
rvec = np.zeros((3, 3), dtype=np.float64)
tvec = np.zeros((3, 1), dtype=np.float64)

# Set flag to EPNP
flag = cv2.SOLVEPNP_EPNP

# Reshape object points and image points
P = np.reshape(P, (6, 1, 3))
p = np.reshape(p, (6, 1, 2))

# Find rotational and translational vector
ret, rvec1, tvec1 = cv2.solvePnP(P, p, M_c, dist_coef, rvec, tvec, flags=flag)

# Convert 3X1 rotation matrix to 3X3 matrix
dst = np.zeros((3, 3), dtype=np.float64)
jacob = np.zeros((3, 9), dtype=np.float64)
dst1, jacob1 = cv2.Rodrigues(rvec1, dst, jacob)

# Matrix M_t that transforms 3D point in world coordinates into the 3D point in the frame of reference of the tracker
result = np.concatenate((dst1, tvec1), axis=1)
result = np.float64(result)
np.set_printoptions(suppress=True)
print("\nMatrix M_t : ")
print(result)

def cal_normal_random(mean, sigma):
    max = 2147483647
    rand_num = []
    for i in range(0, 13):
        x = np.random.randn() / max
        rand_num.append(x)
    c1 = 0.029899776
    c2 = 0.008355968
    c3 = 0.076542912
    c4 = 0.252408784
    c5 = 3.949846138

    r = 0.0

    for i in range(0, 13):
        r += rand_num[i]

    r = (r - 6.0) / 4.0
    r2 = r * r
    gauss_rand = ((((c1 * r2 + c2) * r2 + c3) * r2 + c4) * r2 + c5) * r
    return mean + sigma * gauss_rand

log_arr = []
# x = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
noise_arr = np.zeros(6)
for k in range(1, 20):
    new_g = []
    proj_g = []
    proj_g_noise = []
    v = []
    undo_proj = []

    for i, j in zip(g, range(0, 6)):
       #calculating magnitude and adding noise to each component of gaze vector
        x = [0, 0, 0]
        m = math.sqrt(i[0]*i[0] + i[1]*i[1] + i[2]*i[2])
        sigma = (k*m)/100
        noise = cal_normal_random(0, sigma)
        noise_arr[j] = noise_arr[j] + noise
        x[0] = i[0] + noise_arr[j]
        x[1] = i[1] + noise_arr[j]
        x[2] = i[2] + noise_arr[j]
        new_g.append(x)

    proj_g = np.delete(new_g, np.s_[2], axis=1)
    #another way to do perspective projection of 3D gaze vectore to 2D image points
    for a, b in zip(new_g, proj_g):
        b = np.array(b)/a[2]
        v = a[2]
        b = np.float32(b)
        proj_g_noise.append(b)
    proj_g_noise = np.float64(proj_g_noise)
    proj_g_noise = np.reshape(proj_g_noise, (6, 1, 2))
    ret1, rvec2, tvec2 = cv2.solvePnP(P, proj_g_noise, M_c, dist_coef, rvec, tvec, flags=flag)
    # Convert 3X1 rotation matrix to 3X3 matrix
    dst2, jacob2 = cv2.Rodrigues(rvec2, dst, jacob)

    result1 = np.concatenate((dst2, tvec2), axis=1)
    result1 = np.float64(result1)
    result1 = np.concatenate((result1, [[0, 0, 0, 1]]), axis=0)
    #print("\n Noisy Matrix M_t : ")
    #print(result1)

    M = matrix(result1)
    M_inverse = M.I
    identity = np.dot(M_inverse, M)
    np.set_printoptions(suppress=True)
    x = []
    #undo the projection to find the noisy 3D points
    for l, n in zip(new_g, P):
        l = matrix(l)

        l = np.concatenate((l, [[1]]), axis=1)
        l = np.transpose(l)
        undo_proj = M_inverse.dot(l)

        undo_proj = np.delete(undo_proj, np.s_[3], axis=0)
        n = np.transpose(n)
        h = n - undo_proj
        H = h[0]**2 + h[1]**2 + h[2]**2
        H = H.item()
        x.append(H)
    diff = statistics.mean(x)
    diff_arr.append(diff)
#plotting the graph
y_axis = np.arange(0, 19, 1)
fig, (ax) = plt.subplots(1)
ax.plot(y_axis, diff_arr)
ax.set_title("Reprojection Error Graph", fontsize=20)
ax.set_xlabel("Noise Level (k)", fontsize=16)
ax.set_ylabel("Error", fontsize=16)
# ax.set_xlim([0, 22])
# ax.set_ylim([0, 13])
plt.plot(y_axis, diff_arr)
plt.savefig('error.pdf',bbox_inches='tight')
plt.show()