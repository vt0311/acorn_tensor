'''
Created on 2018. 1. 11.

@author: acorn

파일 이름 : word_tokenize.py

-nltk.tokenize package: 
문장 분할기(분리자)

-word_tokenize(s):
Tokenizers divide strings into lists of substrings.

-관련 사이트 :
http://www.nltk.org/api/nltk.tokenize.html


'''
from nltk.tokenize import word_tokenize

s = 'eee aaa bbb fff'

result = word_tokenize(s)

print(result) # ['eee', 'aaa', 'bbb', 'fff']

print(type(result)) # <class 'list'>

