#!/usr/bin/env python
# Utility for reading text into a one-hot array of bits.

from random import random

import numpy

TERMINATOR = "\0"
CHARSET = " abcdefghijklmnopqrstuvwxyz.:;'\"" + TERMINATOR
CHARACTER_SET = set(CHARSET) # Some naming ambiguity.  Forgive me.
INDEX_CHARACTER_MAP = {k:v for k,v in enumerate(CHARSET)} # index -> letter
CHARACTER_INDEX_MAP = {v:k for k,v in enumerate(CHARSET)} # letter -> index

def get_sentence_vector_length(character_limit):
	return character_limit*len(CHARSET)

def string_to_vector(sentence, flatten=True):
	vector = list()
	for character in sentence.lower():
		subvector = numpy.zeros(len(CHARSET))
		if character in CHARACTER_SET:
			subvector[CHARACTER_INDEX_MAP[character]] = 1.0
		vector.append(subvector)
	result = numpy.asarray(vector)
	if flatten:
		result = result.reshape(1,-1)[0]
	return result

def vector_to_string(vector):
	# Flatten if in square setup.
	if vector.ndim == 2:
		vector = vector.reshape(1,-1)[0]
	s = ""
	# Step through vector in CHARSET increments.
	block_length = len(CHARSET)
	for index in range(0, len(vector), block_length):
		block = vector[index:index+block_length]
		block_energy = block.sum()
		energy = random()*block_energy # This block defines a probability distribution.
		if block_energy == 0:
			continue # No prediction?
			# TODO: Add an invalid char.
		for subindex in range(block_length):
			# Did we randomly select this character?
			if energy < block[subindex]:
				# Yes.
				# Also, is this the end of-line character?
				char = INDEX_CHARACTER_MAP[subindex]
				if char == TERMINATOR:
					return s
				else:
					s += char
					break;
			else:
				energy -= block[subindex]
	return s

