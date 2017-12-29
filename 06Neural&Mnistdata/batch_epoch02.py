'''
Created on 2017. 12. 28.

@author: acorn
'''
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
 
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
 
nb_classes = 10
img_row = 28
img_column = 28
mnistImg = img_row * img_column # 784
 
x = tf.placeholder(tf.float32, [None, mnistImg])
y = tf.placeholder(tf.float32, [None, nb_classes])
 
w = tf.Variable(tf.random_normal([mnistImg, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))
 
H = tf.nn.softmax( tf.matmul(x, w) + b )
 
diff = -tf.reduce_sum( y * tf.log(H), axis = 1 )
cost = tf.reduce_mean( diff )
 
learn_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate= learn_rate)
train = optimizer.minimize( cost )
 
# Test model
prediction = tf.equal( tf.argmax(H, axis = 1) , tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
 
# parameters
training_epochs = 15 # 반복 횟수
batch_size = 100
total_example = mnist.train.num_examples
# total_batch = 55000 / 100 = 550(550번 데이터를 가져 오겠다.)
total_batch = int( total_example / batch_size)
 
print('전체 개수(total_example) : ', total_example) # 전체 갯수(55000)
print('일회 배치 개수(batch_size) : ', batch_size)
print('총 배치 회수(total_batch) : ', total_batch)
print('총 반복 횟수(training_epochs) : ', training_epochs)
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
 
        for i in range(total_batch):
            # 메모리의 용량 때문에 조금씩 배치 한다
            batch_xs, batch_ys = mnist.train.next_batch( batch_size )
            _cost, _train = sess.run([cost, train], feed_dict={
                            x: batch_xs, y: batch_ys})
            avg_cost += _cost / total_batch
 
        print('반복 회수:', '%04d' % (epoch + 1),
              'cost(비용) =', '{:.9f}'.format(avg_cost))
 
    print("학습 종료(Learning finished)")
 
    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, \
            feed_dict={ x : mnist.test.images, y : mnist.test.labels}))
 
    print('\n')
    # 임의의 원소 1개에 대한 예측
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label(정답) : ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction(예측된 데이터): ", sess.run(
        tf.argmax(H, 1), feed_dict={x: mnist.test.images[r:r + 1]}))
 
    # don't know why this makes Travis Build error.
    plt.imshow(
         mnist.test.images[r:r + 1].reshape(img_row, img_column),
         cmap='Greys',
         interpolation='nearest')
    plt.show()