import tensorflow as tf

a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

eq = tf.equal(a, b)

true_condition = {a: 1, b: 1}
false_condition = {a: 1, b: 2}

with tf.Session() as sess:
	print(sess.run(eq, feed_dict = false_condition))