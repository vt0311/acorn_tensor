'''
Created on 2017. 12. 26.

@author: acorn
'''

import tensorflow as tf
import numpy as np

def normalize(input):
    max = np.max(input, axis=0)
    min = np.min(input, axis=0)
    out = (input - min)/(max - min)
    return out

x = [50, 40, 10, 20, 35, 15]

x = normalize(x)

print(x)

print('-----------------------------------------')

x = [[50], [40], [10], [20], [45], [15]]

x = normalize(x)

print(x)

print('----------------------------------------')

x = [[50,15], [40,20], [10,5], [20,10], [45,22], [15,13]]
x = normalize(x)
print( x )
