#!/usr/bin/env python
import sys, os
import csv
import gzip

import numpy
import tensorflow as tf

import bitreader

# We have some big fields.
csv.field_size_limit(sys.maxsize)

# Globals
INPUT_FILENAME = sys.argv[1]
DATA_COLUMN = int(sys.argv[2])
LEARNING_RATE = 0.01
TRAINING_ITERATIONS = 200000
BATCH_SIZE = 1
SENTENCE_LIMIT = 140 # Empty pad after this.
N_INPUTS = bitreader.get_sentence_vector_length(SENTENCE_LIMIT)
WORD_CHUNK_SIZE = 4
DROPOUT = 0.8
DISPLAY_INCREMENT = 100

# Create model
x = tf.placeholder(tf.types.float32, [None, 1, N_INPUTS, 1])
keep_prob = tf.placeholder(tf.types.float32) #dropout

def build_model(name, inputs, dropout_toggle, char_sample_size=WORD_CHUNK_SIZE):
	x = tf.reshape(inputs, shape=[-1, 1, N_INPUTS, 1])

	# conv2d input is [b, h, w, d]
	# filter is [h, w, ...]
	# multiplied by filter [di, dj, :, :]

	sample_vec_len = bitreader.get_sentence_vector_length(char_sample_size)

	wc1 = tf.Variable(tf.random_normal([1, sample_vec_len, 1, 64]))
	bc1 = tf.Variable(tf.random_normal([64,]))
	conv1 = tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME')# + bc1
	act1 = tf.nn.relu(conv1)
	# TensorShape([Dimension(None), Dimension(1), Dimension(4620), Dimension(64)])

	wc2 = tf.Variable(tf.random_normal([1, char_sample_size, 64, 32]))
	bc2 = tf.Variable(tf.random_normal([32,]))
	conv2 = tf.nn.conv2d(act1, wc2, strides=[1, 1, 1, 1], padding='SAME')# + bc2
	act2 = tf.nn.relu(conv2)
	# TensorShape([Dimension(None), Dimension(1), Dimension(4620), Dimension(32)])

	wf1 = tf.Variable(tf.random_normal([sample_vec_len*32, 128]))
	full1 = tf.batch_matmul(act2, wf1)
	act3 = tf.nn.relu(full1)

	encoder = act3

	wf2 = tf.Variable(tf.random_normal([128, sample_vec_len*32]))
	full2 = tf.batch_matmul(act3, wf2)
	act4 = tf.nn.relu(full2)

	wc3 = tf.Variable(tf.random_normal([1, char_sample_size, 32, 64]))
	bc3 = tf.Variable(tf.random_normal([64,]))
	conv3 = tf.nn.conv2d(act4, wc3, strides=[1, 1, 1, 1], padding='SAME')# + bc3
	act5 = tf.nn.relu(conv3)
	# TensorShape([Dimension(None), Dimension(1), Dimension(4620), Dimension(64)])

	wc4 = tf.Variable(tf.random_normal([1, sample_vec_len, 64, 1]))
	bc4 = tf.Variable(tf.random_normal([1,]))
	conv4 = tf.nn.conv2d(act5, wc4, strides=[1, 1, 1, 1], padding='SAME')# + bc4
	act6 = tf.nn.relu(conv4)

	decoder = act6

	return encoder, decoder, [wc1, wc2, wf1, wf2, wc3, wc4], [bc1, bc2, bc3, bc4]

def max_pool(name, l_input, k):
	return tf.nn.max_pool(l_input, ksize=[1, 1, k, 1], strides=[1, 1, k, 1], padding='SAME', name=name)

def norm(name, l_input, lsize):
	return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

encoder, decoder, weights, biases = build_model("ConvNLP", x, keep_prob)

#l2_cost = tf.reduce_mean(tf.nn.l2_loss(reconstruction, x))
l1_cost = tf.reduce_mean(tf.abs(tf.sub(x, decoder)))
cost = l1_cost
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Define our data iterator
def csv_iterator(filename=INPUT_FILENAME):
	if filename.endswith(".gz"):
		fin = gzip.open(filename, 'r')
	else:
		fin = open(filename, 'r')
	cin = csv.reader(fin)
	for line in cin:
		text = line[DATA_COLUMN] # 0 is entry name.
		# Iterate through N characters at a time.
		for index in range(0, len(text), SENTENCE_LIMIT):
			segment = text[index:index+SENTENCE_LIMIT]
			segment = segment + "\0"*(SENTENCE_LIMIT-len(segment)) # Zero-pad our sentence.
			yield bitreader.string_to_vector(segment)

def batch_buffer(filename=INPUT_FILENAME, batch_size=BATCH_SIZE):
	iterator = csv_iterator(filename)
	while True:
		batch = numpy.zeros([batch_size, 1, bitreader.get_sentence_vector_length(SENTENCE_LIMIT), 1], dtype=numpy.float)
		for index, example in zip(range(batch_size), iterator):
			batch[index,0,:,0] = example[:]
		yield batch

# Train
init = tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init)
	step = 1
	generator = batch_buffer()
	for step, batch_xs in zip(range(TRAINING_ITERATIONS/BATCH_SIZE), generator):
		sess.run(optimizer, feed_dict={x: batch_xs, keep_prob: DROPOUT})
		if step % DISPLAY_INCREMENT == 0:
			loss = sess.run(cost, feed_dict={x: batch_xs, keep_prob: 1.})
			enc, rec = sess.run([encoder, decoder], feed_dict={x: batch_xs, keep_prob: 1.})
			print("Iter " + str(step*BATCH_SIZE) + ", Loss= " + "{:.6f}".format(loss))
			print("Example: {} -> {}".format(batch_xs[0], enc))
			print("Example: {} -> {}".format(bitreader.vector_to_string(batch_xs[0,0,:,0]), bitreader.vector_to_string(rec[0,0,:,0])))

		step += 1
	print "Optimization Finished!"
	# sess.run(accuracy, feed_dict={x: asdf, keep_prob: 1.})
	# Save results
	result = saver.save(sess, "result.ckpt")
	print("Saved model to {}".format(result))
