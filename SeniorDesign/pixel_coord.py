import numpy as np
import cv2

img = cv2.imread('./pixel_pic/persp_black.jpg')
background_img = cv2.imread('./pixel_pic/persp_bg.jpg')
# Cannot detect contours of an undistorted image
#newmtx = np.load('./coord_data/newcam_mtx.npy')
#mtx = np.load('./coord_data/cam_mtx.npy')
#dist = np.load('./coord_data/dist.npy')
#img = cv2.undistort(img, mtx, dist, None, mtx)
#backround_img = cv2.undistort(background_img, mtx, dist, None, mtx)

# Convert to grayscale and find the absolute difference between background image and image to be detected
bg_gray = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
diff_gray = cv2.absdiff(bg_gray, im_gray)
diff_blur = cv2.GaussianBlur(diff_gray,(5,5),0) # Blur to smooth edges

# Use a standard otsu thresholding algorithm to wash out the detected objects
ret, otsu_thresh = cv2.threshold(diff_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#May need threshold check here

diff = cv2.GaussianBlur(otsu_thresh,(5,5),0) # Blur to further smooth edges (not necessary)

# Detect contours
im2, contours, hierarchy = cv2.findContours(diff, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
img_cont = img.copy()
cv2.drawContours(img_cont, contours, -1, (0,255,0), 3)
cv2.imwrite('threshold.jpg', diff) # Save thresholded image
cv2.imwrite('contour.jpg', img_cont) # Save contoured image

#Obtain center x and y pixel coordinates of each detected object
cx_arr = []
cy_arr = []
for c in contours:
	M = cv2.moments(c)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	cx_arr.append(cx)
	cy_arr.append(cy)

print(cx_arr)
print(cy_arr)
