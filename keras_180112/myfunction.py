'''
Created on 2018. 1. 12.

@author: acorn
'''

#import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense

def getDataSet(data, testing_row=5):
    # data.shape는 tuple자료형인데, 인덱싱이 가능하다.
    table_row = data.shape[0]
    print(table_row)
    
    #testing_row = 5 # 테스트 용 데이터 셋 개수
    training_row = table_row - testing_row # 훈련용 데이터 셋 개수
    
    # table_col : 엑셀 파일의 컬럼 수
    table_col = data.shape[1] # 열의 갯수
    print(table_col)
    
    y_column = 1
    x_column = table_col - y_column # 입력데이터의 컬럼 갯수
    
    #x_imsi = data[:, 0:x_column]
    #y_imsi = data[:, x_column(x_column+1)]
    
    #print(x_imsi)
    #print(y_imsi)
    
    x_train = data[ 0:training_row, 0:x_column ]
    y_train = data[ 0:training_row, x_column:(x_column+1) ]
    
    x_test  = data[training_row:, 0:x_column ]
    y_test  = data[training_row:, x_column:(x_column+1) ]
    
    return x_train, x_test, y_train, y_test