import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#############################################################################
# 시드를 사용한 동일한 패턴의 랜덤 값 추출
#############################################################################
seed = 3
np.random.seed(seed)
tf.set_random_seed(seed)
#############################################################################
# set batch size for training
batch_size = 100
#############################################################################
# 함수 정의 영역 시작
#############################################################################
# 가중치를 초기화해주는 함수 정의
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(weight)
# 편향을 초기화해주는 함수 정의
def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(bias)
# Create a fully connected layer:
def fully_connected(input_layer, weights, biases):
    # y = x * w + b
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return(tf.nn.relu(layer))
# Normalize by column (min-max norm to be between 0 and 1)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)
#############################################################################
# 함수 정의 영역 끝
#############################################################################
# 엑셀 파일 읽어 들이기
birth_weight_file = 'new_baby.csv' # 189 행 9열

# 즉, 리스트의 요소 갯수가 189행
# birth_data : 각 행은 9열씩 들어 있는 리스트이다.
birth_data = [] 
with open(birth_weight_file, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        # row : 엑셀 파일의 한 행
        birth_data.append(row)
           
# 모든 요소를 float 형으로 변환한다.           
birth_data = [[float(one) for one in row] for row in birth_data]
# print(birth_data)

# birth_data 예시
# [
#     [1.0, 28.0, 113.0, 1.0, 1.0, 1.0, 0.0, 1.0, 709.0], 
#     [1.0, 29.0, 130.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1021.0], 
#     ...
# ]

# y_imsi : 몸무게 정보를 담고 있는 마지막 컬럼
y_imsi = np.array([one[8] for one in birth_data])
# print(y_imsi)
 
# 입력을 위한 데이터의 컬럼 이름
cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI']
 
# x_imsi : 몸무게 컬럼을 제외한 나머지 컬럼(Low 컬럼은 제외)
x_imsi = np.array([[one[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for one in birth_data])
# print(x_imsi)

x_column = 7 # 입력 데이터의 컬럼

# Create Placeholders
x = tf.placeholder(shape=[None, x_column], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
 
#--------Create the first layer (25 hidden nodes)--------
weight_1 = init_weight(shape=[x_column, 25], st_dev=10.0)
bias_1 = init_bias(shape=[25], st_dev=10.0)
layer_1 = fully_connected(x, weight_1, bias_1)
 
#--------Create second layer (10 hidden nodes)--------
weight_2 = init_weight(shape=[25, 10], st_dev=10.0)
bias_2 = init_bias(shape=[10], st_dev=10.0)
layer_2 = fully_connected(layer_1, weight_2, bias_2)
 
#--------Create third layer (3 hidden nodes)--------
weight_3 = init_weight(shape=[10, 3], st_dev=10.0)
bias_3 = init_bias(shape=[3], st_dev=10.0)
layer_3 = fully_connected(layer_2, weight_3, bias_3)
 
#--------Create output layer (1 output value)--------
weight_4 = init_weight(shape=[3, 1], st_dev=10.0)
bias_4 = init_bias(shape=[1], st_dev=10.0)
final_output = fully_connected(layer_3, weight_4, bias_4)

# Split data into train/test = 80%/20%
# 데이터를 80대 20의 비율로 분할한다.
train_indices = np.random.choice(len(x_imsi), round(len(x_imsi)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_imsi))) - set(train_indices)))

x_train = x_imsi[train_indices]
x_test = x_imsi[test_indices]

y_train = y_imsi[train_indices]
y_test = y_imsi[test_indices]
 
x_train = np.nan_to_num(normalize_cols(x_train))
x_test = np.nan_to_num(normalize_cols(x_test))

# Declare cost function (L1)
diff = tf.abs(y - final_output)
cost = tf.reduce_mean(diff)
 
# Declare optimizer
learn_rate = 0.05
optimizer = tf.train.AdamOptimizer( learning_rate = learn_rate)
train = optimizer.minimize(cost)
 
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#############################################################################
# 반복 학습 영역
#############################################################################
loss_vec = []
test_loss = []
for i in range(200):
    rand_index = np.random.choice(len(x_train), size=batch_size)
    rand_x = x_train[rand_index]
    rand_y = np.transpose([y_train[rand_index]])
    sess.run(train, feed_dict={x: rand_x, y: rand_y})
 
    temp_loss = sess.run(cost, feed_dict={x: rand_x, y: rand_y})
    loss_vec.append(temp_loss)
     
    test_temp_loss = sess.run(cost, feed_dict={x: x_test, y: np.transpose([y_test])})
    test_loss.append(test_temp_loss)
    if (i+1) % 25 == 0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))
#############################################################################
# 차트 그리기
#############################################################################
# Plot cost (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
 
actuals = np.array([one[0] for one in birth_data])
test_actuals = actuals[test_indices]
train_actuals = actuals[train_indices]
test_preds = [one[0] for one in sess.run(final_output, feed_dict={x: x_test})]
train_preds = [one[0] for one in sess.run(final_output, feed_dict={x: x_train})]
test_preds = np.array([1.0 if one < 2500.0 else 0.0 for one in test_preds])
train_preds = np.array([1.0 if one < 2500.0 else 0.0 for one in train_preds])
 
test_acc = np.mean([one==yval for one,yval in zip(test_preds, test_actuals)])
train_acc = np.mean([one==yval for one,yval in zip(train_preds, train_actuals)])
 
print('On predicting the category of low birthweight from regression output (<2500g):')
print('Test Accuracy: {}'.format(test_acc))
print('Train Accuracy: {}'.format(train_acc))