import numpy as np
import cv2
import glob

# Create object points with varying x and y but keep z = 0
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)*18


objpoints = [] # 3d points
imgpoints = [] # Pixel Coordinates

# Create array of calibration picture names
images = glob.glob('./LEFT/*.jpg')

# Convert to grayscale for image processing
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find 8x6 chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6),None)

    # If all the corners were found, fill the empty arrays with the points
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria) # increase corner accuracy
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, (8,6), corners2,ret)
        cv2.imshow('img',img) # Check each image to make sure corners were found correctly
        cv2.waitKey(100) # Change for faster or slower viewing

# Calibrate Function
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
cv2.destroyAllWindows()

# Save important matrices
np.save('cam_mtx.npy', mtx)
np.save('dist.npy', dist)
img = cv2.imread('./LEFT/final_left_16.jpg')
h, w = img.shape[:2]

# New Camera Matrix for better undistortion and used for 3-D coordinate transformation
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# Save new values for undistortion
np.save('newcam_mtx.npy', newcameramtx)
np.save('roi.npy', roi)
# undistort
undst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x,y,w,h = roi
# crop with new roi value to check matrix
undst = undst[y:y+h, x:x+w]
cv2.imwrite(filename = 'calibresult.png',img = undst)
