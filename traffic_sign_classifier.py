# Load pickled data
import pickle
import matplotlib.pyplot as plt

import numpy as np

import random

import matplotlib.image as mpimg
import cv2
import numpy
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

X_y_map = zip(X_train, y_train)

labels_plotted = []
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


### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
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
    
    # SOLUTION: Layer 5: Fully Connected. Input = 80. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(80, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
        
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
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y) #logit error across all training labels 

# I am trying to split soft max and cross entrpy functions so that I can pass softmax into top_k

prediction = tf.nn.softmax(logits)
cross_e = - tf.reduce_sum(one_hot_y * tf.log(prediction), reduction_indices = 1)




loss_operation = tf.reduce_mean(cross_entropy) #average cross_entropy
optimizer = tf.train.AdamOptimizer(learning_rate = rate) #variant of Stochastic Gradient Descent
training_operation = optimizer.minimize(loss_operation) #for BackPropagation

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

imgs = ['0.png', '1.png', '2.png', '3.png', '4.png']
y_act = [0, 1, 2, 3, 4]
new_input = []
for imgname in imgs:
    image = mpimg.imread('testSet/' + imgname)
    ch3imag = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    # ch3imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imag32x = cv2.resize(ch3imag, (32,32))
    new_input.append(imag32x)
    # plt.imshow(imag32x)
    # plt.show()
    # plt.savefig("testSet/n_" + imgname)

X_new = numpy.array(new_input)
print("input size: ", X_new.shape)
print("actual labels: ", y_act)

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
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        train_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1), 
              "\tTraining Accuracy = {:.3f}".format(train_accuracy), 
              "\tValidation Accuracy = {:.3f}".format(validation_accuracy))
    print()
    # saver.save(sess, './lenet')
    print("Model saved")
# with tf.Session() as sess:
#     #saver = tf.train.import_meta_graph('./lenet.meta')
#     saver.restore(sess, './lenet')
    train_accuracy = evaluate(X_train, y_train)
    validation_accuracy = evaluate(X_validation, y_validation)
    test_accuracy = evaluate(X_test, y_test)
    print("Training Accuracy = {:.3f}".format(train_accuracy))
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    print("Test Accuracy = {:.3f}".format(test_accuracy))
# Visualizations will be shown in the notebook.
# with tf.Session() as session:
    # saver.restore(session, './lenet')
    google_accuracy = evaluate(new_input,y_act)
    print("Google Data Accuracy = {:.3f}".format(google_accuracy))