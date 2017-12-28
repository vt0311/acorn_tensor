'''
Created on 2017. 12. 28.

@author: acorn
'''
# batch_epoch01.py
from tensorflow.examples.tutorials.mnist import input_data
 
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print( 'type(mnist) : ')
print( type(mnist))
 
img_row = 28 # 이미지 1개의 가로 픽셀 수
img_column = 28 # 이미지 1개의 세로 픽셀 수
 
batch_xs, batch_ys = mnist.train.next_batch(1) # 1개씩 가져 오기
print('type(batch_xs) :', type(batch_xs)) # <class 'numpy.ndarray'>
# print('batch_xs.reshape( img_row, img_column ) :', batch_xs.reshape( img_row, img_column ))
print('batch_xs.shape :', batch_xs.shape) # shape(1, 784)
print('batch_ys :', batch_ys) #  레이블 보기
print('batch_ys.shape :', batch_ys.shape)
 
plt.imshow(batch_xs.reshape( img_row, img_column ), cmap='Greys')
plt.show()



# 출력 결과
# type(mnist) : 
# <class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>
# type(batch_xs) : <class 'numpy.ndarray'>
# batch_xs.shape : (1, 784)
# batch_ys : [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
# batch_ys.shape : (1, 10)
