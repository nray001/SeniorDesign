import cv2
import numpy as np

# Load data and undistort
roi = np.load('./coord_data/roi.npy')
newmtx = np.load('./coord_data/newcam_mtx.npy')
mtx = np.load('./coord_data/cam_mtx.npy')
dist = np.load('./coord_data/dist.npy')
# Load with object filled image (Image to be used with IK)
img = cv2.imread('./pixel_pic/objects.jpg')
img = cv2.undistort(img, mtx, dist, None, newmtx)
# cannot crop because matrix was made using original undistorted image
#x,y,w,h = roi
#img = img[y:y+h, x:x+w]
cv2.imwrite('object_undst.jpg', img)

# Here are the center coordinates of the camera found in the new camera matrix
cx = 407 #[0,2] (Matrix indexes)
cy = 391 #[1,2] (Matrix indexes)

# Put a point at the center of the frame (camera centerpoint)
cv2.circle(img,(cx,cy),3,(0,255,0),2)
while True:
	cv2.imshow('img', img) # View image to find pixel coordinates of object
	key = cv2.waitKey(1)
	if key == ord('q'):
		break

cv2.destroyAllWindows()
