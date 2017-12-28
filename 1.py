'''
题目：有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？
程序分析：可填在百位、十位、个位的数字都是1、2、3、4。组成所有的排列后再去 掉不满足条件的排列。
'''

#!/usr/bin/python
#-*- coding: UTF-8 -*-
'''
# 方法一
d = []
for i in range(1, 5):
    for j in range(1, 5):
        for k in range(1, 5):
            if( i != k ) and (i != j) and (j != k):
                d.append([i, j, k])
print("总数量：", len(d))
print(d)

#方法二
list_num = [1, 2, 3, 4]

list = [i*100 + j*10 + k for i in list_num for j in list_num for k in list_    num if(j != i and k != i and k != j)]

print(list)

#方法三
from itertools import permutations

for i in permutations([1, 2, 3, 4], 3):
    print(i)
'''

# 方法四：没事找事之位运算
# 从 00 01 10 到 11 10 01
for num in range(6, 58):
    a = num >> 4 & 3
    b = num >> 2 & 3
    c = num & 3
    if( (a^b) and (b^c) and (c^a) ):
        print(a+1, b+1, c+1)
