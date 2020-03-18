import numpy as np
import cv2
import glob
import serial

# Load data for calculations
cam_mtx = np.load('./cam_mtx.npy')
dist = np.load('./dist.npy')
newcam_mtx = np.load('./newcam_mtx.npy')
roi = np.load('./roi.npy')

# Load center x,y and focal length from new camera matrix
cx = newcam_mtx[0,2]
cy = newcam_mtx[1,2]
fx = newcam_mtx[0,0]

# Real world coordinates (cm) of center camera point
X_center = 23.5
Y_center = 21.9
Z_center = 32.39

# Filled with real world coordinates of perspective points (measured with a ruler in cm) 
worldPoints = np.array([[X_center, Y_center, Z_center],
[4.13,3.18,41.59],[13.97,3.18,36.83],[23.81,3.18,35.56],[4.13,10.8,39.1],
[13.97,10.8,34.93],[23.81,10.8,32.7],[4.13,18.42,36.83],[13.97,18.42,33],[23.81,18.42,31.43]], dtype=np.float32)

#Filled with image coordinated respective to real world coordinates
imagePoints = np.array([[cx, cy],
[331,322],[369,321],[409,321],[332,352],[371,351],[408,351],
[334,380],[371,380],[408,380]], dtype=np.float32)

for i in range(1,10):
	# calculate approximate z distance from camera lens using trig (pythag. theorem)
	wX = worldPoints[i,0]- X_center
	wY = worldPoints[i,1]- Y_center
	wd = worldPoints[i,2]
	d1 = np.sqrt(np.square(wX) + np.square(wY))
	wZ = np.sqrt(np.square(wd) - np.square(d1))
	worldPoints[i,2] = wZ

# calculate and save data used for coordinate transformation
inverse_newcam_mtx = np.linalg.inv(newcam_mtx)
np.save('inverse_newcam_mtx.npy', inverse_newcam_mtx)
# obtain calibrated rvec and tvec
ret, rvec1, tvec1 = cv2.solvePnP(worldPoints,imagePoints,newcam_mtx,dist)
# Use Rodrigues function to obtain rotation matrix
R_mtx, jac = cv2.Rodrigues(rvec1)
np.save('R_mtx.npy', R_mtx)
np.save('rvec1.npy', rvec1) # Calibrated rotation vector
np.save('tvec1.npy', tvec1) # Calibrated translation vector


# Scaling factor array
s_arr=np.array([0], dtype=np.float32)
s_describe=np.array([0,0,0,0,0,0,0,0,0,0],dtype=np.float32)

# Calculate percent errors in calibrated tvec and rvec as well as scaling factor for increased accuracy
# (Received directly from Perspective Calibration reference)

for i in range(0,10):
	print("=======POINT # " + str(i) +" =========================")
	print("Forward: From World Points, Find Image Pixel")
	XYZ1=np.array([[worldPoints[i,0],worldPoints[i,1],worldPoints[i,2],1]], dtype=np.float32)
	XYZ1=XYZ1.T
	print("{{-- XYZ1")
	print(XYZ1)
	suv1=P_mtx.dot(XYZ1)
	print("//-- suv1")
	print(suv1)
	s=suv1[2,0]    
	uv1=suv1/s
	print(">==> uv1 - Image Points")
	print(uv1)
	print(">==> s - Scaling Factor")
	print(s)
	s_arr=np.array([s/10+s_arr[0]], dtype=np.float32)
	s_describe[i]=s

	print("Solve: From Image Pixels, find World Points")

	uv_1=np.array([[imagePoints[i,0],imagePoints[i,1],1]], dtype=np.float32)
	uv_1=uv_1.T
	print(">==> uv1")
	print(uv_1)
	suv_1=s*uv_1
	print("//-- suv1")
	print(suv_1)

	print("get camera coordinates, multiply by inverse Camera Matrix, subtract tvec1")
	xyz_c=inverse_newcam_mtx.dot(suv_1)
	xyz_c=xyz_c-tvec1
	print("      xyz_c")
	inverse_R_mtx = np.linalg.inv(R_mtx)
	XYZ=inverse_R_mtx.dot(xyz_c)
	print("{{-- XYZ")
	print(XYZ)

s_mean, s_std = np.mean(s_describe), np.std(s_describe)

print(">>>>>>>>>>>>>>>>>>>>> S RESULTS")
print("Mean: "+ str(s_mean))
print("Average: " + str(s_arr[0])) #scaling factor
print("Std: " + str(s_std))

print(">>>>>> S Error by Point")

np.save('s_arr.npy', s_arr) #save scaling array to obtain scaling factor (s_arr[0])

for i in range(0,10):
	print("Point "+str(i))
	print("S: " +str(s_describe[i])+" Mean: " +str(s_mean) + " Error: " + str(s_describe[i]-s_mean))
