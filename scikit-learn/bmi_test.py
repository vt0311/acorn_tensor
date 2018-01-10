'''
Created on 2018. 1. 10.

@author: acorn

키와 몸무게 데이터를 이용하여 SVM으로 학습시켜 보고, 비만을 정확하게 맞출 수 있는지 테스트
해보도록 한다.
• 파일 이름 : bmi_test.py, bmi.csv
'''
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd

# 키와 몸무게 데이터 읽어 들이기 --- (※1)
tbl = pd.read_csv("bmi.csv")

# 칼럼(열)을 자르고 정규화하기 --- (※2)
label = tbl["label"]

w = tbl["weight"] / 100 # 최대 100kg라고 가정
h = tbl["height"] / 200 # 최대 200cm라고 가정
wh = pd.concat([w, h], axis=1)

# 학습 전용 데이터와 테스트 전용 데이터로 나누기 --- (※3)
x_train, x_test, y_train, y_test = \
    train_test_split(wh, label)
    
# 데이터 학습하기 --- (※4)
clf = svm.SVC()

clf.fit(x_train, y_train)

# 데이터 예측하기 --- (※5)
predict = clf.predict(x_test)

# 결과 테스트하기 --- (※6)
ac_score = metrics.accuracy_score(y_test, predict)

print("정답률 =", ac_score)

cl_report = metrics.classification_report(y_test, predict)

print("\n리포트 =\n", cl_report)