#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Tensors (쿡북 29쪽) 텐서 정의
#----------------------------------
#
# This function introduces various ways to create
# tensors in TensorFlow

import tensorflow as tf
# 모델(그래프) 초기화
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Introduce tensors in tf

# Get graph handle
sess = tf.Session()

# 0으로 채워진 텐서 생성(1행 20열)
my_tensor = tf.zeros([1,20])
#print(my_tensor)
print('my_tensor:', sess.run(my_tensor))

# Declare a variable
my_var = tf.Variable(tf.zeros([1,20]))
print(sess.run(my_var))

# Different kinds of variables
row_dim = 2
col_dim = 3 

# Zero initialized variable (2행 3열을 0으로 초기화)
zero_var = tf.Variable(tf.zeros([row_dim, col_dim]))
#print('zero_var:', sess.run(zero_var))

# One initialized variable (2행 3열을 1로 초기화)
ones_var = tf.Variable(tf.ones([row_dim, col_dim]))

# shaped like other variable
sess.run(zero_var.initializer)
sess.run(ones_var.initializer)

# 기존 텐서를 이용하여 초기화하기
zero_similar = tf.Variable(tf.zeros_like(zero_var))
ones_similar = tf.Variable(tf.ones_like(ones_var))

# session 영역에 개별 변수를 초기화할 때는 initializer를 사용하면 된다.
sess.run(ones_similar.initializer)
sess.run(zero_similar.initializer)

# Fill shape with a constant (2행 3열을 생성하되 모두 -1로 채운다.)
fill_var = tf.Variable(tf.fill([row_dim, col_dim], -1))

# Create a variable from a constant
const_var = tf.Variable(tf.constant([8, 6, 7, 5, 3, 0, 9]))
# This can also be used to fill an array:
const_fill_var = tf.Variable(tf.constant(-1, shape=[row_dim, col_dim]))

# Sequence generation
# Generates [0.0, 0.5, 1.0] includes the end
# 시작점과 종료점을 포함하여 요소 3개를 만들어라.(2등분)
linear_var = tf.Variable(tf.linspace(start=0.0, stop=1.0, num=3)) # Generates [0.0, 0.5, 1.0] includes the end

# Generates [6, 9, 12] doesn't includes the end
# 6부터 시작하여 3씩 증가하되 15직전까지
sequence_var = tf.Variable(tf.range(start=6, limit=15, delta=3)) # Generates [6, 9, 12] doesn't include the end

# Random Numbers

# Random Normal
# 평균이 0.0이고 표준편차가 1.0인 랜덤한 값을 2행 3열 생성한다.
rnorm_var = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)

# Add summaries to tensorboard
merged = tf.summary.merge_all()

# Initialize graph writer:

writer = tf.summary.FileWriter("/tmp/variable_logs", graph=sess.graph)

# Initialize operation
initialize_op = tf.global_variables_initializer()

# Run initialization of variable
sess.run(initialize_op)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++