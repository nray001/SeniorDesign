import numpy as np
import cv2

#Load important camera data and calibrated vectors for transformation
R_mtx = np.load('./coord_data/R_mtx.npy')
tvec1 = np.load('./coord_data/tvec1.npy')
inverse_newcam_mtx = np.load('./coord_data/inverse_newcam_mtx.npy')
s_arr = np.load('./coord_data/s_arr.npy')
# Ask for cx and cy (u,v)
u = input("Enter u: ")
v = input("Enter v: ")

# Perform linear matrix algebra based off of coordinate equation
inverse_R_mtx = np.linalg.inv(R_mtx)
uv_1 = np.array([[u,v,1]], dtype = np.float32)
uv_1 = uv_1.T
suv_1 = s_arr[0] * uv_1 # s_arr[0] = scaling factor
xyz_c = inverse_newcam_mtx.dot(suv_1)
xyz_c = xyz_c - tvec1
xyz = inverse_R_mtx.dot(xyz_c)
# Scale coordinates to fit our IK coordinate system
x = xyz[0]
y = xyz[1]
#x = 23.18 - x # 23.18 cm = horizontal distance from origin
#y = 15.24 + y # 15.24 cm = vertical distance from origin
# Convert to inches (IK function uses inches)
x = x / 2.54
y = y / 2.54
x = np.round(x,1)
y = np.round(y,1)
print(x)
print(y)
