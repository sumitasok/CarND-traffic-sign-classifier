import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

imgs = ['0.png', '1.png', '2.png', '3.png', '4.png']
y_act = [0, 1, 2, 3, 4]
new_input = []

fig, axes = plt.subplots(2, 5, figsize=(32, 32),
                         subplot_kw={'xticks': [], 'yticks': []})

fig.subplots_adjust(hspace=0.3, wspace=0.05)

for ax in axes.flat:
    for imgname in imgs:
        image = mpimg.imread('testSet/' + imgname)
        ax.imshow(image)
        # ax.set_title("l: " +str(label) + " c: " + str(label_dict[label]))
        ax.set_xticks([])
        ax.set_yticks([])

    for imgname in imgs:
        image = mpimg.imread('testSet/' + imgname)
        # plt.figure()
        # plt.imshow(image)
        # plt.savefig("testSet/same-"+imgname)
        resized = cv2.resize(image, (32, 32))
        # plt.figure()
        # plt.imshow(resized)
        # plt.savefig("testSet/reshaped-" + imgname)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        new_input.append(gray)
        print(imgname, 'image is of size:', gray.shape)
        # plt.imshow(gray, cmap='gray')
        # plt.show()
        # plt.savefig("testSet/gray-" + imgname)

        ax.imshow(gray, cmap='gray')
        # ax.set_title("l: " +str(label) + " c: " + str(label_dict[label]))
        ax.set_xticks([])
        ax.set_yticks([])

plt.savefig("testSet/input_normalisation.jpg")