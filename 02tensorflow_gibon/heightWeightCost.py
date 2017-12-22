print()
import tensorflow as tf

x = [65, 80, 90, 45]
y = [165, 190, 160, 185 ]

w = tf.Variable(1.0)
b = tf.Variable(1.0)

H = w * x + b

diff = tf.square(H-y)
cost = tf.reduce_mean(diff)

