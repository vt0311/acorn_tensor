print()
import tensorflow as tf

x = [65, 80, 90, 45]
y = [165, 190, 160, 185 ]

w = tf.Variable(1.0)
b = tf.Variable(1.0)

H = w * x + b

diff = tf.square(H-y)
cost = tf.reduce_mean(diff)

learn_rate = 1e-4
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
train= optimizer.minimize(cost)

sess = tf.Session() 

sess.run(tf.global_variables_initializer())
 
for step in range(10000):
    sess.run(train)
    print('step : %d, cost : %.12f, weight : %f, bias : %f' % \
        ( step, sess.run(cost), sess.run(w), sess.run(b)))

