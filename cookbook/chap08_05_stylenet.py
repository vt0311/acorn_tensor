'''
Created on 2018. 1. 17.

@author: acorn
'''
# Using TensorFlow for Stylenet/NeuralStyle
#---------------------------------------
#
# We use two images, an original image and a style image
# and try to make the original image in the style of the style image.
#
# Reference paper:
# https://arxiv.org/abs/1508.06576
#
# Need to download the model 'imagenet-vgg-verydee-19.mat' from:
#   http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat

# 바로 위의 인터넷 주소를 복사하고, 해당 파일을 현재 폴더에 복사하시오.

# Stylenet : 스타일 이미지를 학습하여 원본 이미지를 스타일 이미지의 스타일로 적용시키는 기법

import scipy.io
import scipy.misc
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()


# Image Files
original_image_file = './images/book_cover.jpg' # 원본 파일
style_image_file = './images/starry_night.jpg' # 스타일 파일

# Saved VGG Network path under the current project dir.
# imagenet-vgg-19 신경망
# vgg_path : 사전에 미리 학습했던 신경망 파일
vgg_path = 'imagenet-vgg-verydeep-19.mat' # 바이너리 파일

# mat 파일 --> scipy --> python 


# Default Arguments
original_image_weight = 5.0  # 원본 이미지의 최초 가중치
style_image_weight = 500.0  # 스타일 이미지의 최초 가중치
regularization_weight = 100
learning_rate = 0.001  # 학습율
generations = 5000
#generations = 60  # 총 학습 횟수
#output_generations = 250
output_generations = 20  # 20회마다 출력하겠다. 

# 아담 옵티마이져 관련 변수
beta1 = 0.9 # 1번째 moment에 대한 지수형 감쇠기
beta2 = 0.999 # 2번째 monent에 대한 지수형 감쇠기

# Read in images. scipy를 이용한 이미지 로딩
original_image = scipy.misc.imread(original_image_file)
style_image = scipy.misc.imread(style_image_file)

# Get shape of target and make the style image the same
target_shape = original_image.shape

# imresize : 원본 이미지의 크기와 동일하게 스타일 이미지를 리사이징
style_image = scipy.misc.imresize(style_image, target_shape[1] / style_image.shape[1])

# VGG-19 Layer Setup
# From paper(논문)
# 논문 저자의 명명법에 따라서, 다음과 같이 계층(layer)을 정의한다.
# len(vgg_layers)는 36이다.
vgg_layers = ['conv1_1', 'relu1_1',
              'conv1_2', 'relu1_2', 'pool1',
              'conv2_1', 'relu2_1',
              'conv2_2', 'relu2_2', 'pool2',
              'conv3_1', 'relu3_1',
              'conv3_2', 'relu3_2',
              'conv3_3', 'relu3_3',
              'conv3_4', 'relu3_4', 'pool3',
              'conv4_1', 'relu4_1',
              'conv4_2', 'relu4_2',
              'conv4_3', 'relu4_3',
              'conv4_4', 'relu4_4', 'pool4',
              'conv5_1', 'relu5_1',
              'conv5_2', 'relu5_2',
              'conv5_3', 'relu5_3',
              'conv5_4', 'relu5_4']

# Extract weights and matrix means
# mat 파일에서 매개변수를 추출해주는 함수
def extract_net_info(path_to_params):
    vgg_data = scipy.io.loadmat(path_to_params)
    normalization_matrix = vgg_data['normalization'][0][0][0]
    mat_mean = np.mean(normalization_matrix, axis=(0,1)) # 평균
    network_weights = vgg_data['layers'][0] # 가중치
    return(mat_mean, network_weights) # 튜플로 리턴
    

# Create the VGG-19 Network
# 가중치 및 계층 정의로부터 tf 신경망을 재구축해주는 함수
def vgg_network(network_weights, init_image):
    # 매개변수
    # network_weights : 사전 학습망에서 구한 가중치 정보
    # init_image : 원본이미지의 placeholder 정보
    network = {}
    image = init_image

    for i, layer in enumerate(vgg_layers):
        if layer[0] == 'c':
            weights, bias = network_weights[i][0][0][0][0]
            weights = np.transpose(weights, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            conv_layer = tf.nn.conv2d(image, tf.constant(weights), (1, 1, 1, 1), 'SAME')
            image = tf.nn.bias_add(conv_layer, bias)
        elif layer[0] == 'r':
            image = tf.nn.relu(image)
        else:
            image = tf.nn.max_pool(image, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
        network[layer] = image
    # network 는 다음과 같은 형식으로 데이터가 들어 있을 것이다.
    # network = {'conv1_1':'cnn수행결과1', 'relu1_1':'cnn수행결과2', ...}    
    return(network)

# Here we define which layers apply to the original or style image
# 원본 이미지에 적용시킬 레이어
original_layer = 'relu4_2'
# 스타일 이미지에 적용시킬 레이어
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

# Get network parameters
# normalization_mean : 평균
# network_weight : 가중치
normalization_mean, network_weights = extract_net_info(vgg_path)

# tf의 이미지의 연산은 4차원(rank : 4)이다.
# 앞에 차원 1개 추가하기 위하여 (1,)을 붙인다.
shape = (1,) + original_image.shape
style_shape = (1,) + style_image.shape
original_features = {}
style_features = {}

# Get network parameters
image = tf.placeholder('float', shape=shape)
vgg_net = vgg_network(network_weights, image)

# Normalize original image
# 이미지를 정규화하고, 신경망을 통하여 실행시킨다.
original_minus_mean = original_image - normalization_mean
original_norm = np.array([original_minus_mean])
original_features[original_layer] = sess.run(vgg_net[original_layer],
                                             feed_dict={image: original_norm})

# Get style image network
# 스타일 이미지를 위한 플레이스 홀더 지정
image = tf.placeholder('float', shape=style_shape)
vgg_net = vgg_network(network_weights, image)
# 스타일 이미지에 대한 정규화
style_minus_mean = style_image - normalization_mean
style_norm = np.array([style_minus_mean])

# 신경망 실행
for layer in style_layers:
    layer_output = sess.run(vgg_net[layer], feed_dict={image: style_norm})
    layer_output = np.reshape(layer_output, (-1, layer_output.shape[3]))
    style_gram_matrix = np.matmul(layer_output.T, layer_output) / layer_output.size
    style_features[layer] = style_gram_matrix

# Make Combined Image
# 결합 이미지를 생성하기 위하여 잡음 이미지를 신경망에 투입하여 실행한다.
# shape : 원본 이미지의 shape
initial = tf.random_normal(shape) * 0.256
image = tf.Variable(initial)
vgg_net = vgg_network(network_weights, image)

# Loss (교재 391쪽)
# l2_loss : sum(t ** 2)/2 (제곱의 총합 / 2)
original_loss = original_image_weight * (2 * tf.nn.l2_loss(vgg_net[original_layer] - original_features[original_layer]) /
                original_features[original_layer].size)
                
# Loss from Style Image
# 각각의 스타일 이미지에 대한 비용 계산
style_loss = 0
style_losses = []

# style_layers : 스타일 이미지에 적용시킬 계층 정보
for style_layer in style_layers:
    # vgg_net[style_layer] : 연산이 이미 되어 있는(rank:4)
    layer = vgg_net[style_layer]
    feats, height, width, channels = [x.value for x in layer.get_shape()]
    size = height * width * channels
    features = tf.reshape(layer, (-1, channels))
    style_gram_matrix = tf.matmul(tf.transpose(features), features) / size
    style_expected = style_features[style_layer]
    style_losses.append(2 * tf.nn.l2_loss(style_gram_matrix - style_expected) / style_expected.size)

# style_image_weight : 스타일 이미지의 가중치
style_loss += style_image_weight * tf.reduce_sum(style_losses)
        
# To Smooth the resuts, we add in total variation loss       
total_var_x = sess.run(tf.reduce_prod(image[:,1:,:,:].get_shape()))
total_var_y = sess.run(tf.reduce_prod(image[:,:,1:,:].get_shape()))
first_term = regularization_weight * 2
second_term_numerator = tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:])
second_term = second_term_numerator / total_var_y
third_term = (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) / total_var_x)

# total_varation_loss : 전체 변이 비용
# 교재 392쪽
# 깨끗한 이미지는 변이값이 낮고,
# 잡음이 많은 이미지는 변이값이 높다.
total_variation_loss = first_term * (second_term + third_term)
    
# Combined Loss
# 손실 = 원본계층비용 + 스타일계층비용 + 전체변이비용
loss = original_loss + style_loss + total_variation_loss

# Declare Optimization Algorithm
optimizer = tf.train.AdamOptimizer(learning_rate,beta1,beta2)
train_step = optimizer.minimize(loss)

# Initialize Variables and start Training
# 세션 초기화
sess.run(tf.global_variables_initializer())
# generations : 총 학습 횟수
for i in range(generations):

    sess.run(train_step)

    # Print update and save temporary output
    if (i+1) % output_generations == 0:
        print('Generation {} out of {}, loss: {}'.format(i + 1, generations,sess.run(loss)))
        image_eval = sess.run(image)
        best_image_add_mean = image_eval.reshape(shape[1:]) + normalization_mean
        # 중간 과정의 임시 이미지 저장하기
        output_file = 'temp_output_{}.jpg'.format(i)
        scipy.misc.imsave(output_file, best_image_add_mean)
        
        
# Save final image
image_eval = sess.run(image)
best_image_add_mean = image_eval.reshape(shape[1:]) + normalization_mean
# 최종 이미지 파일
output_file = 'final_output.jpg'
# scipy : science 관련 (수학, 과학)
scipy.misc.imsave(output_file, best_image_add_mean)
