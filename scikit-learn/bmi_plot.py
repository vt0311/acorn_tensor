'''
Created on 2018. 1. 10.

@author: acorn

일반적으로 데이터의 분포는 산포도를 그려보면 알 수 있다.
• BMI 공식을 이용하여 데이터를 만들었으므로 당연히 분포가 잘 되었겠지만 이를 그림으로 한 번
그려 보도록 한다.
• 파일 이름 : bmi_plot.py

'''

import matplotlib.pyplot as plt
import pandas as pd

# Pandas로 CSV 파일 읽어 들이기
tbl = pd.read_csv("bmi.csv", index_col=2)

# 그래프 그리기 시작
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# 서브 플롯 전용 - 지정한 레이블을 임의의 색으로 칠하기
def scatter(lbl, color):
    b = tbl.loc[lbl]
    ax.scatter(b["weight"],b["height"], c=color, label=lbl)
    
scatter("fat", "red")
scatter("normal", "yellow")
scatter("thin", "purple")
ax.legend()

plt.savefig("bmi-test.png")
plt.show()