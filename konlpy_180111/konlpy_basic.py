'''
Created on 2018. 1. 11.

@author: acorn
'''
from konlpy.tag import Twitter
# Twitter를 이용하여 twitter 객체를 생성한다.
twitter = Twitter()
text = '아버지 가방에 들어 가신다. 그래욕ㅋㅋ'
# pos() : 형태소를 분석해준다.
# norm 매개 변수 : 그래욕ㅋㅋ → 그래요
# stem 매개 변수 : 그래욕ㅋㅋ → 그렇다.
malist = twitter.pos( text, norm=True, stem = True )
print( malist )