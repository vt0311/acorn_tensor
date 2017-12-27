'''
Created on 2017. 12. 27.

@author: acorn
'''
import numpy as np

def inlineprint(mylist):
    imsi = ''
    for item in mylist :
        imsi += str(item) + ' '
    print( '[', imsi, ']' )
    
H = np.array([[0.5], [0.6], [0.7], [0.8] ])

print('일반적인 출력 형식')
print(H)
print('\n인라인 출력하기')
inlineprint(H)
    