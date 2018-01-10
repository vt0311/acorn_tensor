'''
Created on 2018. 1. 10.

@author: acorn
''' 

import urllib.request as req
local= "mushroom.csv"

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

req.urlretrieve(url, local)

print("ok")

