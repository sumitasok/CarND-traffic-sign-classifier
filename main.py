# Load pickled data
import pickle
import matplotlib.pyplot as plt
# %matplotlib inline


import numpy as np

import random

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

index = random.randint(0, len(X_train))




import pandas as pd

signMapDf = pd.read_csv("signnames.csv")

label_samples = {}
counter = 1
for _feature, _label in zip(train['features'], train['labels']):
    if _label in label_samples:
        continue
    # counter is designed to quit the loop if all the required csv labels are identified.
    if counter == 43:
        break
    counter += 1
    label_samples[_label] = {"image": _feature, "label": _label, "signname": signMapDf.loc[_label].SignName}









def normalize(images, newMax=1., newMin=0., oldMax=255, oldMin=0):
    return (images - oldMin) * ((newMax - newMin)/(oldMax - oldMin)) + newMin

N_TRANSFORMS = 6
SEED = 7

# https://pypi.python.org/pypi/tqdm
from tqdm import tqdm_notebook, tnrange, tqdm
from keras.preprocessing.image import ImageDataGenerator
def add_augumented_data(x, y, nb_transforms, rotation = 30, trans_range = 0.3, channel_shift = 0.05):
    datagen = ImageDataGenerator(rotation_range=rotation,                           
                                 channel_shift_range=channel_shift,
                                 shear_range=trans_range,
                                 zoom_range=trans_range,
                                 width_shift_range=trans_range,
                                 height_shift_range=trans_range,
                                 fill_mode='nearest')

    datagen.fit(x, seed=SEED)
    batch_size = x.shape[0]

    for i in tqdm(range(nb_transforms), desc='Jittering'):       
        X_batch, y_batch = next(datagen.flow(x, y, batch_size=batch_size, seed=SEED))

        x = np.concatenate((x, X_batch), axis=0)
        y = np.concatenate((y, y_batch), axis=0)
        
    return x,y


# grayer = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# grayfunc = np.vectorize(grayer)
# type(X_train)
# grayfunc(X_train)

# def grayscale(images):
#     x = np.ndarray((32, 32))
#     for image in images:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         x = np.concatenate((x, gray), axis=0)
#     return x


X_train = normalize(X_train).astype(np.float32)
X_valid = normalize(X_valid).astype(np.float32)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, 
#                                                   y_train,
#                                                   test_size=0.3,
#                                                   random_state=SEED)

# print("data augmentation started")
# X_train, y_train = add_augumented_data(X_train, y_train, N_TRANSFORMS)
# print("data augmentation completed")

print("Number of training features =", X_train.shape[0])
print("Number of validation features =", X_valid.shape[0])
print("Number of test features =", X_test.shape[0])


n = 5
k = 5

def plot_top_k(n, k, top_5_samples, X_new, y_act, label_samples):
    fig, ax = plt.subplots(n, k + 1, figsize=(20, 20),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for idx, kresults in enumerate(top_5_samples):
        print("idx : ", idx, " actual : ", y_act[idx], " kresults : ", kresults)
        # ax[idx][0].imshow(X_new[idx].squeeze())
        # ax[idx][0].set_title(str(y_act[idx]) + ' :' + label_samples[y_act[idx]]['signname'])
        # for rIdx, result in enumerate(kresults):
            # ax[idx][rIdx+1].imshow(label_samples[result]['image'].squeeze())
            # color = 'b'
            # if result == y_act[idx]:
            #     color = 'r'
            # ax[idx][rIdx+1].set_title(str(result) + ' :' + label_samples[result]['signname'], color=color)
            # ax[idx][rIdx+1].set_xticks([])
            # ax[idx][rIdx+1].set_yticks([])

# X_train = grayscale(X_train)
# X_valid = grayscale(X_valid)
# X_test = grayscale(X_test)
# print("done")
# print(len(grayscale(X_train)))


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)



### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

# X_train = normalize(X_train).astype(np.float32)
# X_test = normalize(X_test).astype(np.float32)

# X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
X_validation, y_validation = X_valid, y_valid
X_train, y_train = shuffle(X_train, y_train)

EPOCHS = 10
BATCH_SIZE = 128




from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional Activation. Input = 32x32x3. Output = 28x28x10.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 10), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(10))
    conv1   = tf.nn.relu(tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b)

    # SOLUTION: Pooling. Input = 28x28x10. Output = 14x14x10.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional Activation. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 10, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.relu(tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected Activation. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.nn.relu(tf.matmul(fc0, fc1_W) + fc1_b)
    
    # SOLUTION: Layer 4: Fully Connected Activation. Input = 120. Output = 80.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120,80), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(80))
    fc2    = tf.nn.relu(tf.matmul(fc1, fc2_W) + fc2_b)
    
    # SOLUTION: Layer 5: Fully Connected. Input = 80. Output = 60.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(80, 60), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(60))
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b

    # SOLUTION: Layer 5: Fully Connected. Input = 80. Output = 43.
    fc4_W  = tf.Variable(tf.truncated_normal(shape=(60, 43), mean = mu, stddev = sigma))
    fc4_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc3, fc4_W) + fc4_b


        
    return logits



### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

rate = 0.001  #learning rate tells how quickly to update network weights

logits = LeNet(x)  #pass input to LeNet to calc

prediction = tf.nn.softmax(logits)
top_5 = tf.nn.top_k(prediction, k=5, sorted=True)
cross_e = - tf.reduce_sum(one_hot_y * tf.log(prediction), reduction_indices = 1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y) #logit error across all training labels 
loss_operation = tf.reduce_mean(cross_entropy) #average cross_entropy
optimizer = tf.train.AdamOptimizer(learning_rate = rate) #variant of Stochastic Gradient Descent
training_operation = optimizer.minimize(loss_operation) #for BackPropagation

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    # X_data = normalize(X_data).astype(np.float32)
    num_examples = len(X_data)
    total_accuracy = 0
    top_5_list = []
    e_logits_list = []
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy, top_5_x, e_logits = sess.run([accuracy_operation, top_5, logits], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        top_5_list.append(top_5_x)
        e_logits_list.append(e_logits)
    return total_accuracy / num_examples, top_5_list, e_logits_list

import time
timestamp = str(int(time.time()*1000000))
print("timestamp", timestamp)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run([training_operation, top_5, logits], feed_dict={x: batch_x, y: batch_y})
            
        # X_train = normalize(X_train)
        # X_validation = normalize(X_validation)
        train_accuracy, train_top_5, tr_logits = evaluate(X_train, y_train)
        validation_accuracy, validate_top_5, val_logits= evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1), 
              "\tTraining Accuracy = {:.3f}".format(train_accuracy), 
              "\tValidation Accuracy = {:.3f}".format(validation_accuracy))
    print()
    saver.save(sess, './lenet/'+timestamp)
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, './lenet/'+timestamp)
    # train_accuracy, train_top_5, tr_logits = evaluate(X_train, y_train)
    # validation_accuracy, validation_top_5, val_logits = evaluate(X_validation, y_validation)
    test_accuracy, test_top_5, test_logits = evaluate(X_test, y_test)
    print("Training Accuracy = {:.3f}".format(train_accuracy))
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# print(timestamp)
def probs_from_logits(logits, idx):
    return logits[idx]
# print(probs_from_logits(test_logits[0],0))


print("---------- First run -------------------")

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import matplotlib.image as mpimg
import cv2
import numpy
# import scipy as sp

# Visualizations will be shown in the notebook.
# %matplotlib inline
imgs = ['0.png', '1.png', '2.png', '3.png', '4.png']
y_act = [0, 1, 2, 3, 4]
new_input = []
f, ax = plt.subplots(1, 5, figsize=(40, 40))
for idx, imgname in enumerate(imgs):
    image = mpimg.imread('testSet/' + imgname)
    ch3imag = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    imag32x = cv2.resize(ch3imag, (32,32))
    ax[idx].imshow(imag32x)
    new_input.append(imag32x)

    
    
X_new = numpy.array(new_input)
print("input size: ", X_new.shape)
print("actual labels: ", y_act)


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
with tf.Session() as session:
    saver.restore(session, './lenet/'+timestamp)
    print("start evaluation")
    new_accuracy, top_5_samples, e_logits = evaluate(X_new,y_act)
    print("done evaluation with accuracy: ", new_accuracy)

plot_top_k(n,k, top_5_samples[0][1], X_new, y_act, label_samples)

print(top_5_samples[0][1])
print(new_accuracy)
print(len(X_new))
print(e_logits[0][0])
print(np.argmax(e_logits[0][0]))


print("---------- Second run -------------------")



imgs = ['sign-3.png', 'sign-12.png', 'sign-14.png', 'sign-30.png', 'sign-34.png']
y_act2 = [3, 12, 14, 35, 34]
new_input = []
f, ax = plt.subplots(1, 5, figsize=(40, 40))
for idx, imgname in enumerate(imgs):
    image = mpimg.imread('testSet/' + imgname)
#     ch3imag = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    imag32x = cv2.resize(image, (32,32))
    ax[idx].imshow(imag32x)    
    new_input.append(imag32x)

    
    
X_new2 = numpy.array(new_input)
print("input size: ", X_new2.shape)
print("actual labels: ", y_act2)



with tf.Session() as session:
    saver.restore(session, './lenet/'+timestamp)
    print("start evaluation")
    new_accuracy2, top_5_samples2, e_logits2 = evaluate(X_new2,y_act2)
    print("done evaluation with accuracy: ", new_accuracy2)

plot_top_k(n,k, top_5_samples2[0][1], X_new2, y_act2, label_samples)

# print(top_5_samples[0][1])


