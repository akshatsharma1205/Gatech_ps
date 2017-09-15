import cv2
import numpy as np 

sample = cv2.imread("/Users/john/Desktop/football.png", 0)

def conv_transform(image):
	image_copy = image.copy()

	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			image_copy[i][j] = image[image.shape[0]-i-1][image.shape[1]-j-1]
	return image_copy

def conv(image, kernel):
	# The image will be grayscale, otherwise there will be confusion with 3 channels
	kernel = conv_transform(kernel)
	image_h = image.shape[0]
	image_w = image.shape[1]

	kernel_h = kernel.shape[0]
	kernel_w = kernel.shape[1]

	h = kernel_h//2#integer
	w = kernel_w//2

	image_conv = np.zeros(image.shape)

	for i in range(h, image_h-h):
		for j in range(w, image_w-w):
			sum = 0

			for m in range(kernel_h):
				for n in range(kernel_w):
					sum = (sum + kernel[m][n]*image[i-h+m][j-w+n])

			image_conv[i][j] = sum
	#cv2.imshow('Convolved_image', image_conv)
	
	return image_conv
	
	'''		
	cv2.imwrite('/Users/john/ps/assop/pilo1_conv_rgb.jpg', image_conv)
	cv2.waitKey(0)
	cv2.destroyALlWindows()
	'''
# SOBEL_FIELDMAN EDGE
def norm(img1, img2):
	img_copy = np.zeros(img1.shape)# image with intitial zero values
	#img_copy = img1.copy()

	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
			q = (img1[i][j]**2 + img2[i][j]**2)**(1/2)
			if(q>120):# Threshold
				img_copy[i][j] = 255 #obtaining a binary image
			else:
				img_copy[i][j] = 0
			
	return img_copy

kernel = np.zeros(shape=(3,3))
kernel[0, 0] = -1
kernel[0, 1] = -2
kernel[0, 2] = -1
kernel[1, 0] = 0
kernel[1, 1] = 0
kernel[1, 2] = 0
kernel[2, 0] = 1
kernel[2, 1] = 2
kernel[2, 2] = 1
gy =	conv(sample, kernel)
cv2.imshow("gradient_y", gy)

kernel[0, 0] = -1
kernel[0, 1] = 0
kernel[0, 2] = 1
kernel[1, 0] = -1
kernel[1, 1] = 0
kernel[1, 2] = 1
kernel[2, 0] = -2
kernel[2, 1] = 0
kernel[2, 2] = 2
gx =	conv(sample, kernel)
cv2.imshow("gradient_x", gx)
g_sobel = norm(gx, gy)

cv2.imshow("Sobel_edge", g_sobel)
#cv2.imwrite("/Users/john/ps/assop/sobel_try.png", g_sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()


#OTSU's thresholding-- 



