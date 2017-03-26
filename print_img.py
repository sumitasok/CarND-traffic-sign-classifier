# Load pickled data
import pickle
import matplotlib.pyplot as plt
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

# index = random.randint(0, len(X_train))

# for i in range(0, len(X_train)):
# image = X_train[index].squeeze()
# plt.figure(figsize=(1,1))
# plt.imshow(image, cmap="gray")
# plt.savefig(str(index)+".png")
# print(y_train[index])

print("Number of training examples {}", len(X_train))
print("Number of testing examples = ", len(X_test))
print("Image data shape =", X_train[0].shape)
print("All Training labels = ", y_train)
print("Number of classes =", np.unique(y_train, return_counts = True))
