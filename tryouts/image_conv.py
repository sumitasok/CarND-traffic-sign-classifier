import matplotlib.image as mpimg
import cv2
import numpy

# Visualizations will be shown in the notebook.
imgs = ['0.png', '1.png', '2.png', '3.png', '4.png']
y_act = [0, 1, 2, 3, 4]
new_input = []
for imgname in imgs:
    image = mpimg.imread('testSet/' + imgname)
    new_input.append(image)
    print(imgname, "image shape", image.shape)
    ch3imag = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    print(imgname, "image shape after BGRA2BGR", ch3imag.shape)
    imag32x = cv2.resize(ch3imag, (32,32))
    print(imgname, "image shape resize 32", imag32x.shape)
    # plt.imshow(image)
    # plt.show()

X_new = numpy.array(new_input)
print("input size: ", X_new.shape)
print("actual labels: ", y_act)