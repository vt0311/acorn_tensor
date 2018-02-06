import os
import re
# import io
# import requests
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
# from zipfile import ZipFile
# from tensorflow.python.framework import ops
# ops.reset_default_graph()

sess = tf.Session()

# Set RNN parameters
# epochs = 20
# batch_size = 250
max_sequence_length = 7
# rnn_size = 10
embedding_size = 50
min_word_frequency = 10
# learning_rate = 0.0005
# dropout_keep_prob = tf.placeholder(tf.float32)

data_dir = 'temp'
data_file = 'text_imsi.txt'

text_data = []
with open(os.path.join(data_dir, data_file), 'r') as file_conn:
    for row in file_conn:
        text_data.append(row)
text_data = text_data[:-1]

text_data = [x.split('\t') for x in text_data if len(x)>=1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]

def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+\n', '\n', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return(text_string)
 
# Clean texts
text_data_train = [clean_text(x) for x in text_data_train]

print('text_data \n', text_data)
print('text_data_target \n', text_data_target)
print('text_data_train \n', text_data_train)

# Change texts into numeric vectors
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,
                                                                     min_frequency=min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))
 
# Shuffle and split data
text_processed = np.array(text_processed)
print('text_processed\n', text_processed)

text_data_target = np.array([1 if x=='ham' else 0 for x in text_data_target])
print('text_data_target\n', text_data_target)

print('len(text_data_target) : \n', len(text_data_target))
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]
 
# Split train/test set
ix_cutoff = int(len(y_shuffled)*0.80)

x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]

vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))
 
# Create placeholders
x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])
 
# Create embedding
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
print('embedding_mat\n', embedding_mat)

embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)
print('embedding_output\n', embedding_output)