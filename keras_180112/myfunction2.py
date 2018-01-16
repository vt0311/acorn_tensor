'''
Created on 2018. 1. 12.

@author: acorn
'''
from keras.utils import np_utils

#import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense

def getDataSet(data, testing_row=5, one_hot = False, num_classes=-1):
    # data.shape는 tuple자료형인데, 인덱싱이 가능하다.
    # one_hot : 원핫인코딩 여부 지정, False 이면 하지 않겠다.
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
    
    if one_hot == False :
    #    y_train = data[0:training_row, x_column:(x_column+1)] 
        pass
    else : # one hot 인코딩이 필요한 경우
        if num_classes >= 1:
            y_train = np_utils.to_categorical(y_train, num_classes)
    
    x_test  = data[training_row:, 0:x_column ]   
    y_test  = data[training_row:, x_column:(x_column+1) ]
    
    return x_train, x_test, y_train, y_test



def getDataProp(data, testing_rate=0.2, one_hot = False, num_classes=-1):
    # data.shape는 tuple자료형인데, 인덱싱이 가능하다.
    # one_hot : 원핫인코딩 여부 지정, False 이면 하지 않겠다.
    table_row = data.shape[0]-1
    print('table_row:', table_row)
    
    testing_row =  round(table_row * 0.2) # 테스트 용 데이터 셋 개수
    print('testing_row:', testing_row )
    training_row = table_row - testing_row # 훈련용 데이터 셋 개수
    #training_row = table_row * 0.8
    print('training_row:', training_row)
    
    # table_col : 엑셀 파일의 컬럼 수
    table_col = data.shape[1] # 열의 갯수
    print('table_col:', table_col)
    
    y_column = 1
    x_column = table_col - y_column # 입력데이터의 컬럼 갯수
    
    
    x_train = data[ 0:training_row, 0:x_column ]
    y_train = data[ 0:training_row, x_column:(x_column+1) ]
    
    if one_hot == False :
        pass
    else : # one hot 인코딩이 필요한 경우
        if num_classes >= 1:
            y_train = np_utils.to_categorical(y_train, num_classes)
    
    x_test  = data[training_row:, 0:x_column ]   
    y_test  = data[training_row:, x_column:(x_column+1) ]
    
    return x_train, x_test, y_train, y_test