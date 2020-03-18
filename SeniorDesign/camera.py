import cv2
import os

# Default port for initial camera device
left = cv2.VideoCapture("/dev/video0")

# Indexes for timer and picture count
i = 0
j = 0

while(True):
	check, fullFrame = left.read()
	# Crop stereo image to obtain only the left lens frame
	# Horizontal aspect of leftframe is cropped more than it should be to reduce distortion at edges (roi bug)
	leftFrame = fullFrame[0:239, 30:290]
	cv2.imshow("Frame", leftFrame)
	key = cv2.waitKey(1)
	j = j + 50 # Timer for camera screenshots
	if j == 5500:
		j = 0
		# Scale calibration images to find chessboard corners easier
		scale_percent = 200 # percent of original size
		width = int(260 * scale_percent / 100)
		height = int(240 * scale_percent / 100)
		dim = (width, height)
		resized_left = cv2.resize(leftFrame, dim)
		# Save images and relocate them to LEFT folder
		cv2.imwrite(filename='left_img_%s.jpg' % i, img=resized_left)
		os.rename('/home/pi/SeniorDesign/left_img_%s.jpg' % i, '/home/pi/SeniorDesign/LEFT/final_left_%s.jpg' % i)
		print("Image saved")
		i = i+1
		cv2.destroyAllWindows()
		# Save 60 images for increased camera matrix accuracy
		if i >= 60:
			print("Captures Complete")
			break
