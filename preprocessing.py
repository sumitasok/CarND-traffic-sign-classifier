from PIL import Image
# from cv2 import getPerspectiveTransform, warpPerspective
import cv2
import numpy as np

def hello():
    print('hello')

# crop the image using the margin format that keras.cropping2D uses.
# makes it simpler to port the cropping configurations.
# https://keras.io/layers/convolutional/#cropping2d
# http://matthiaseisen.com/pp/patterns/p0202/
def crop_like_keras_crop2D(input_filename, output_filename, top_crop, bottom_crop, left_crop, right_crop):
    img = Image.open(input_filename)
    x_length, y_length = img.size
    cropped_image = img.crop((left_crop, top_crop, x_length - right_crop, y_length - bottom_crop))
    cropped_image.save(output_filename)
    img.close()
    return output_filename

#   src = np.float32([
#        [850, 320],
#        [865, 450],
#        [533, 350],
#        [535, 210]
#    ])
#   src = np.float32([
#        [870, 240],
#        [870, 370],
#        [520, 370],
#        [520, 240]
#    ])
def warp(img, src_points, dst_points, img_size=None):

    if img_size == None:
        img_size = (img.shape[1], img.shape[0])

    M = cv2.getPerspectiveTransform(src_points, dst_points)

    Minv = cv2.getPerspectiveTransform(dst_points, src_points)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.max(gradmag)/255
    gradmag = (gradmag/scaled_sobel).astype(np.uint8) 
    # 5) Create a binary mask where mag thresholds are met
    binary_sobel = np.zeros_like(gradmag)
    binary_sobel[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_sobel

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply cv2.Sobel()
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    if orient == 'x':
        # Take the absolute value of the output from cv2.Sobel()
        abs_sobel = np.absolute(sobelx)
    else:
        abs_sobel = np.absolute(sobely)
    # Scale the result to an 8-bit range (0-255)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Apply lower and upper thresholds
    thresh_min = 20
    thresh_max = 100
    # Create binary_output
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary

def image_process(img):
    a = np.array(img.getdata()).astype(np.float32).reshape( (img.size[0],img.size[1],3) )
    cropped = cv2.resize(a[60:140,:], (320, 80))
    
    R = cropped[:,:,0]
    G = cropped[:,:,1]
    B = cropped[:,:,2]
    thresh = (200, 255)
    rbinary = np.zeros_like(R)
    gbinary = np.zeros_like(G)
    rbinary[(R > thresh[0]) & (R <= thresh[1])] = 1
    
    return np.dstack((rbinary, gbinary, gbinary))