import cv2
import numpy as np 


# Difference between correlation and convolution
def conv_transform(image):
	image_copy = image.copy()

	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			image_copy[i][j] = image[image.shape[0]-i-1][image.shape[1]-j-1]
	return image_copy

def conv(image, kernel):
	# The image will be grayscale, otherwise there will be confusion with 3 channels
	kernel = conv_transform(kernel)# to differ it from corelation
	image_h = image.shape[0]#7
	image_w = image.shape[1]#7

	kernel_h = kernel.shape[0]#3
	kernel_w = kernel.shape[1]#3

	h = kernel_h//# integer value
	w = kernel_w//2

	image_conv = np.zeros(image.shape)

	for i in range(h, image_h-h):
		for j in range(w, image_w-w):
			sum = 0

			for m in range(kernel_h):
				for n in range(kernel_w):
					sum = sum + kernel[m][n]*image[i-h+m][j-w+n]

			image_conv[i][j] = sum
	#cv2.imshow('Convolved_image', image_conv)
	
	return image_conv
	
	'''		
	cv2.imwrite('/Users/john/ps/assop/pilo1_conv_rgb.jpg', image_conv)
	cv2.waitKey(0)
	cv2.destroyALlWindows()
	'''

