def spam_func(x):
    mylist = ['광고', '18']
    result = x in mylist
    if result :
        return '선정성, 광고성 메일'
    else :    
        return '일반 메일'

txt = '광고'
print(spam_func(txt))

txt = '호호'
print(spam_func(txt))
