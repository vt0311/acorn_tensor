'''
Created on 2017. 12. 27.

@author: acorn
'''
import matplotlib.pyplot as plt
import numpy as np

# w를 1이라고 가정
def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

x = np.arange( -5.0, 5.0, 0.1 )

y = sigmoid(x)

print(y)

plt.plot(x,y)

plt.ylim(-0.1, 1.1)

plt.show()
