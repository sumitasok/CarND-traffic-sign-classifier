# Load pickled data
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#%matplotlib inline
import tensorflow as tf
import numpy as np

import random

EPOCHS = 10
BATCH_SIZE = 128

# TODO: Fill this in based on where you saved the training and testing data

training_file = "dataset/train.p"
validation_file= "dataset/valid.p"
testing_file = "dataset/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# for i in range(0, len(X_train)):
# image = X_train[index].squeeze()
# plt.figure(figsize=(1,1))
# plt.imshow(image, cmap="gray")
# plt.savefig(str(index)+".png")
# print(y_train[index])

# print("Number of training examples {}", len(X_train))
# print("Number of testing examples = ", len(X_test))
# print("Image data shape =", X_train[0].shape)
# print("All Training labels = ", y_train)
# print("Number of classes =", np.unique(y_train, return_counts = True))

X_y_map = zip(X_train, y_train)
# uniq_X_y, X_y_count = np.unique(X_y_map, return_counts = True)

# gs = gridspec.GridSpec(8, 3, top=1., bottom=0., right=1., left=0., hspace=0.,
#         wspace=0.)

# uniq_X = np.unique(X_train)

labels_plotted = []
# for g in gs:
#     ax = plt.subplot(g)
#     for img, label in X_y_map:
#         if label not in labels_plotted:
#             print("caught ", label, "current", labels_plotted)
#             ax.imshow(img.squeeze())
#             ax.set_xticks([])
#             ax.set_yticks([])
#             labels_plotted.append(label)
#             break


# np.random.seed(0)
# grid = np.random.rand(4, 4)

label_dict = {}

label_count = np.unique(y_train, return_counts = True)

for label_count_item in zip(label_count[0], label_count[1]):
    label_dict[label_count_item[0]] = label_count_item[1]

fig, axes = plt.subplots(8, 6, figsize=(12, 12),
                         subplot_kw={'xticks': [], 'yticks': []})

fig.subplots_adjust(hspace=0.3, wspace=0.05)

for ax in axes.flat:
    for img, label in X_y_map:
        if label not in labels_plotted:
            ax.imshow(img.squeeze())
            ax.set_title("l: " +str(label) + " c: " + str(label_dict[label]))
            ax.set_xticks([])
            ax.set_yticks([])
            labels_plotted.append(label)
            break

# plt.show()

plt.savefig("sample.jpg")