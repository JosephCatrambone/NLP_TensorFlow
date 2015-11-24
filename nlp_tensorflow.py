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
TRAINING_ITERATIONS = 20000000
BATCH_SIZE = 5
SENTENCE_LIMIT = 140 # Empty pad after this.
N_INPUTS = bitreader.get_sentence_vector_length(SENTENCE_LIMIT)
WORD_CHUNK_SIZE = 4
DROPOUT = 0.8
DISPLAY_INCREMENT = 1

# Create model
x = tf.placeholder(tf.types.float32, [None, 1, N_INPUTS, 1])
keep_prob = tf.placeholder(tf.types.float32) #dropout

def build_model(name, inputs, dropout_toggle, char_sample_size=WORD_CHUNK_SIZE):
	#x = tf.reshape(inputs, shape=[-1, 1, N_INPUTS, 1])

	filter_bank_0 = 16
	filter_bank_1 = 8
	filter_bank_3 = 64

	# conv2d input is [b, h, w, d]
	# filter is [h, w, ...]
	# multiplied by filter [di, dj, :, :]

	sample_vec_len = bitreader.get_sentence_vector_length(char_sample_size)

	wc1 = tf.Variable(tf.random_normal([1, sample_vec_len, 1, filter_bank_0]))
	bc1 = tf.Variable(tf.random_normal([filter_bank_0,]))
	conv1 = tf.nn.conv2d(inputs, wc1, strides=[1, 1, 1, 1], padding='SAME') + bc1
	act1 = tf.nn.relu(conv1)
	# TensorShape([Dimension(None), Dimension(1), Dimension(4620), Dimension(64)])

	wc2 = tf.Variable(tf.random_normal([1, char_sample_size, filter_bank_0, filter_bank_1]))
	bc2 = tf.Variable(tf.random_normal([filter_bank_1,]))
	conv2 = tf.nn.conv2d(act1, wc2, strides=[1, 1, 1, 1], padding='SAME') + bc2
	act2 = tf.nn.relu(conv2)
	norm2 = tf.nn.lrn(act2, bitreader.get_sentence_vector_length(1), bias=1.0, alpha=0.001, beta=0.75)
	# TensorShape([Dimension(None), Dimension(1), Dimension(4620), Dimension(32)])

	# Conv -> FC
	# Record encoder shapes for later use.
	act2_shape = act2.get_shape().as_list()
	act1_shape = act1.get_shape().as_list()
	input_shape = inputs.get_shape().as_list()

	# Resize
	c_fc = tf.reshape(act2, [-1, act2_shape[1]*act2_shape[2]*act2_shape[3]])

	# FC segments
	wf1 = tf.Variable(tf.random_normal([act2_shape[1]*act2_shape[2]*act2_shape[3], filter_bank_3]))
	bf1 = tf.Variable(tf.random_normal([filter_bank_3,]))
	full1 = tf.matmul(c_fc, wf1) + bf1
	act3 = tf.nn.relu(full1)

	# Our prized encoder.
	encoder = act3

	# Invert steps and begin decoder.
	# Start with FC.
	wf2 = tf.Variable(tf.random_normal([filter_bank_3, act2_shape[1]*act2_shape[2]*act2_shape[3]]))
	bf2 = tf.Variable(tf.random_normal([act2_shape[1]*act2_shape[2]*act2_shape[3],]))
	full2 = tf.matmul(act3, wf2) + bf2
	act4 = tf.nn.relu(full2)

	# FC -> Conv
	fc_c = tf.reshape(act4, [-1, act2_shape[1], act2_shape[2], act2_shape[3]])

	wc3 = tf.Variable(tf.random_normal([1, char_sample_size, filter_bank_0, filter_bank_1]))
	bc3 = tf.Variable(tf.random_normal([act1_shape[1], act1_shape[2], act1_shape[3]]))
	conv3 = tf.nn.deconv2d(fc_c, wc3, strides=[1, 1, 1, 1], padding='SAME', output_shape=[-1, act1_shape[1], act1_shape[2], act1_shape[3]]) + bc3
	act5 = tf.nn.relu(conv3)
	# TensorShape([Dimension(None), Dimension(1), Dimension(4620), Dimension(64)])

	wc4 = tf.Variable(tf.random_normal([1, sample_vec_len, 1, filter_bank_0]))
	bc4 = tf.Variable(tf.random_normal([input_shape[1], input_shape[2], input_shape[3]]))
	conv4 = tf.nn.deconv2d(act5, wc4, strides=[1, 1, 1, 1], padding='SAME', output_shape=[-1, input_shape[1], input_shape[2], input_shape[3]]) + bc4
	act6 = tf.nn.relu(conv4)
	norm3 = tf.nn.lrn(act6, bitreader.get_sentence_vector_length(1), bias=1.0, alpha=0.001, beta=0.75)

	decoder = norm3

	return encoder, decoder, [wc1, wc2, wf1, wf2, wc3, wc4], [bc1, bc2, bc3, bc4]

	#return tf.nn.max_pool(l_input, ksize=[1, 1, k, 1], strides=[1, 1, k, 1], padding='SAME', name=name)

print("Building model.")
encoder, decoder, weights, biases = build_model("ConvNLP", x, keep_prob)

print("Defining loss functions and optimizer.")
#l2_cost = tf.reduce_sum(tf.nn.l2_loss(reconstruction, x))
l1_cost = tf.reduce_sum(tf.abs(tf.sub(x, decoder)))
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
print("Gathering variables.")
init = tf.initialize_all_variables()
saver = tf.train.Saver()

print("Beginning training session.")
with tf.Session() as sess:
	print("Initializing variables.")
	sess.run(init)
	step = 1
	generator = batch_buffer()
	print("Session, variables, and generator initialized.  Training.")
	for step, batch_xs in zip(range(TRAINING_ITERATIONS/BATCH_SIZE), generator):
		sess.run(optimizer, feed_dict={x: batch_xs, keep_prob: DROPOUT})
		if step % DISPLAY_INCREMENT == 0:
			loss = sess.run(cost, feed_dict={x: batch_xs, keep_prob: 1.})
			enc, rec = sess.run([encoder, decoder], feed_dict={x: batch_xs, keep_prob: 1.})
			print("Iter " + str(step*BATCH_SIZE) + ", Loss= " + "{:.6f}".format(loss))
			print("Mapping {} -> {}".format(enc.shape, rec.shape))
			print("Example: {} -> {}".format(bitreader.vector_to_string(batch_xs[0,0,:,0]), bitreader.vector_to_string(rec[0,0,:,0])))
		step += 1
	print "Optimization Finished!"
	# sess.run(accuracy, feed_dict={x: asdf, keep_prob: 1.})
	# Save results
	result = saver.save(sess, "result.ckpt")
	print("Saved model to {}".format(result))
