import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import preprocessing as pp
import math
import glob

base_path = '/Users/sumitasok/ml_data/Self-Driving-Car/Behavioural-Cloning/data/'

subject_images = [
    'center_2016_12_01_13_30_48_287.jpg',
    'center_2016_12_01_13_31_12_937.jpg',
    'center_2016_12_01_13_32_35_588.jpg',
    'center_2016_12_01_13_32_46_689.jpg', # approaching the bridge
    'center_2016_12_01_13_32_55_684.jpg', # a normal road turning right at the end / scratching tarred road
    'center_2016_12_01_13_33_04_080.jpg', # tarred road, nearing right turn
    'center_2016_12_01_13_33_06_005.jpg', # tarred road, with read and white striped borders
    'center_2016_12_01_13_33_10_377.jpg', # red-white border, changing to concrete border.
    'center_2016_12_01_13_33_37_309.jpg', # no border on left; right turn
    'center_2016_12_01_13_33_40_049.jpg', # concrete block on left side, while exiting no border.
    'center_2016_12_01_13_33_40_861.jpg', # right side white border
    'center_2016_12_01_13_33_53_864.jpg', # red-white stripe left turn
    'center_2016_12_01_13_33_57_413.jpg', # red-white border changing to concrete border
    'center_2016_12_01_13_34_06_043.jpg', # left no border, concrete block, red-white border on right, turning right
    'center_2016_12_01_13_34_18_784.jpg', # kacha bridge road exiting to normal road.
]

lines = []
with open(base_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

def image_process(current_path):
    image = mpimg.imread(current_path)

    cropped = cv2.resize(image[60:140,:], (320, 80))
    
    R = cropped[:,:,0]
    G = cropped[:,:,1]
    B = cropped[:,:,2]
    thresh = (200, 255)
    rbinary = np.zeros_like(R)
    gbinary = np.zeros_like(G)
    rbinary[(R > thresh[0]) & (R <= thresh[1])] = 1
    
    return np.dstack((rbinary, gbinary, gbinary))

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    # print(type(img))
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=7):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # print("Lines ", line_img.shape, lines)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # Grab the x and y sizes and make two copies of the image
    # With one copy we'll extract only the pixels that meet our selection,
    # then we'll paint those pixels red in the original image to see our selection 
    # overlaid on the original.
    ysize = image.shape[0]
    xsize = image.shape[1]
    color_select= np.copy(image)
    line_image = np.copy(image)

    gray_image = np.copy(image)
    gray = grayscale(gray_image)

    kernel_size = 7
    # print(str(type(gray)))
    blur_gray = gaussian_blur(gray, kernel_size)
    # image_output = gaussian_blur(image_output, 5)

    # Define our parameters for Canny and apply
    # low_threshold = 97
    # high_threshold = 130
    low_threshold = 80
    high_threshold = 160
    # low_threshold = 65
    # high_threshold = 195
    # edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # cv2.imwrite("./test_edges.jpg", edges)

    # print(str(type(blur_gray)))
    # print(str(type(low_threshold)))

    edges = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)

    # vertices = np.array([[(70, 539),(465, 317), (495, 317), (900, 539)]], dtype=np.int32)
    # masked_edges = region_of_interest(edges, vertices)


    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 24     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 1 #minimum number of pixels making up a line
    max_line_gap = 325  # maximum gap in pixels between connectable line segments
    # line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    # print("Hough Lines", lines)

    line_image = hough_lines(edges, rho, theta, threshold, min_line_length, max_line_gap)
    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_select, 1, line_image, 1, 0)
    # masked_edges = region_of_interest(lines_edges, vertices)

    return lines_edges

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

f, ax = plt.subplots(6, 2, figsize=(40, 20))

lines = [
'testSet/0.png',
'testSet/2.png',
'testSet/4.png',
'testSet/sign-14.png',
'testSet/sign-30.png',
'testSet/1.png',
'testSet/3.png',
'testSet/sign-12.png',
'testSet/sign-3.png',
'testSet/sign-34.png',

]

print(lines)

for line in lines:
    filename = line.split('/')[-1]
    print(line)
    # image = cv2.imread(current_path)
    image = mpimg.imread(line)
    # image = image_process(current_path)

    # crop the image

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    # get Red channel
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    rthresh = (200, 255)
    rbinary = np.zeros_like(R)
    rbinary[(R > rthresh[0]) & (R <= rthresh[1])] = 1
    gbinary = np.zeros_like(G)

    # image2 = np.dstack((rbinary, gbinary, gbinary))

    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = image[:,:,0]
    L = image[:,:,1]
    S = image[:,:,2]

    sthresh = (200, 255)
    sbinary = np.zeros_like(S)
    sbinary[(S > sthresh[0]) & (S <= sthresh[1])] = 1

    ksize = 15
    sobely = pp.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh_min=100, thresh_max=200)

    # gaussianBlur = cv2.GaussianBlur(bgr_image, (ksize+4, ksize+4), 0)
    blurring_ksize = 3
    gaussianBlur = cv2.GaussianBlur(sobely, (blurring_ksize, blurring_ksize), 0)

    cannyThresh = (45, 135)
    # canny = cv2.Canny(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY), cannyThresh[0], cannyThresh[1])


    wide = cv2.Canny(gaussianBlur, 10, 200)
    tight = cv2.Canny(gaussianBlur, 200, 250)
    auto = auto_canny(gaussianBlur)

    stacked = np.dstack((wide, tight, auto))

    # write to file

    fontsize=20

    ax[0, 0].imshow(image)
    ax[0, 0].set_title("original RGB", fontsize=fontsize)
    ax[0, 1].imshow(gray_image, cmap='gray')
    ax[0, 1].set_title("Red channel", fontsize=fontsize)
    ax[1, 0].imshow(sbinary)
    ax[1, 0].set_title("Saturation Channel", fontsize=fontsize)
    ax[1, 1].imshow(sobely)
    ax[1, 1].set_title("Sobel Y", fontsize=fontsize)
    ax[2, 0].imshow(gray_image)
    ax[2, 0].set_title("Gray Image", fontsize=fontsize)
    ax[2, 1].imshow(gaussianBlur)
    ax[2, 1].set_title("Gaussian Blur", fontsize=fontsize)
    # ax[3, 0].imshow(canny)
    ax[3, 0].set_title("Canny threshold "+str(cannyThresh[0])+" "+str(cannyThresh[1]), fontsize=fontsize)
    # ax[3, 1].imshow(process_image(bgr_image))
    ax[4, 0].imshow(auto_canny(gaussianBlur))
    ax[4, 0].set_title('auto canny', fontsize=fontsize)
    ax[4, 1].imshow(wide)
    ax[4, 1].set_title('wide', fontsize=fontsize)
    ax[5, 0].imshow(tight)
    ax[5, 0].set_title('tight', fontsize=fontsize)
    ax[5, 1].imshow(stacked)
    ax[5, 1].set_title('stacked', fontsize=fontsize)

    f.savefig('results/videos/'+filename+'.png')
    # plt.close(f)





    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('img', image)
    # cv2.waitKey(500)

cv2.destroyAllWindows()
