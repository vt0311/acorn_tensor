'''
Created on 2017. 12. 27.

@author: acorn
'''
import numpy as np

x = [2.0, 1.0, 0.1 ]
y = []

for item in range(len(x)) :
    y.append(np.exp(x[item]))
    
mysum = np.sum(y)
print('총합 : ', mysum)

total = 0.0
for item in range(len(y)) :
    total = total + y[item] / mysum
    print(y[item] / mysum)
    
print('확률 종합 : ', total )    