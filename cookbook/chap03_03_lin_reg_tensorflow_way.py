# Linear Regression: TensorFlow Way (쿡북 118)
#----------------------------------
# 텐서 플로우 선형 회귀 방식
#
# This function shows how to use TensorFlow to
# solve linear regression.
# y = Ax + b
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Petal Width

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data]) # 꽃잎의 너비 (Petal Width)
y_vals = np.array([y[0] for y in iris.data]) # 꽃받침의 길이 (Sepal Length)

# Declare batch size
batch_size = 25 # 한번 훈련시 25씩 훈련시키겠다.

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[1,1])) # weight
b = tf.Variable(tf.random_normal(shape=[1,1])) # bias

# Declare model operations
model_output = tf.add(tf.matmul(x_data, A), b)

# Declare loss function (L2 loss)
loss = tf.reduce_mean(tf.square(y_target - model_output))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = [] # 비용 함수를 저장할 리스트
for i in range(100):
    # 25개씩 복원 추출
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    
    rand_x = np.transpose([x_vals[rand_index]]) # shape(25, 1)
    rand_y = np.transpose([y_vals[rand_index]]) # shape(25, 1)
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    
    loss_vec.append(temp_loss) # 비용 함수들을 리스트에 저장
    
    if (i+1)%25==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

# Get the optimal coefficients
[slope] = sess.run(A)  # 기울기
[y_intercept] = sess.run(b)  # y절편

# Get best fit line
best_fit = []  # 그래프로 그릴 최적의 직선 정보 리스트
for i in x_vals:
  best_fit.append(slope*i+y_intercept)

# Plot the result
# 꽃잎의 너비(x)와 꽃받침의 길이(y)를 점 그래프로 그리기
plt.plot(x_vals, y_vals, 'o', label='Data Points')

# 꽃잎의 너비에 따른 최적화된 직선 그리기
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)

plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.show()
