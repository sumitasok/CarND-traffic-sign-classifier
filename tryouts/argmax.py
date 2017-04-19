import tensorflow as tf

def printArgmax(x, axis):
	with tf.Session() as sess:
		output = sess.run(tf.argmax(x, axis))
		print(output)

printArgmax([1,1,1,1], 0) # 0
printArgmax([0,1,1,1], 0) # 1
printArgmax([[0,1,0,0], [0,0,1,1], [0,0,0,1], [0,0,0,1]], 1) # [1, 2, 3, 3]