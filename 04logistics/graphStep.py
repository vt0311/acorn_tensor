'''
Created on 2017. 12. 22.

@author: acorn
'''
import numpy as np
import matplotlib.pylab as plt
from numpy.core.multiarray import dtype

def step_function(x):
    return np.array(x>0, dtype = np.int)

x = np.arange(-5.0, 5.0, 0.1)

y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)

plt.show()
