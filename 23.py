'''
题目：打印出如下图案（菱形）:
   *
  ***
 *****
*******
 *****
  ***
   *
程序分析：先把图形分成两部分来看待，前四行一个规律，后三行一个规律，利用双重for循环，第一层控制行，第二层控制列。
程序源代码：
'''

  1 #!/usr/bin/python
  2 #-*- coding: UTF-8 -*-
  3
  4 from sys import stdout
  5 for i in range(4):
  6     for j in range(2 - i + 1):
  7         stdout.write(' ')
  8     for k in range(2 * i + 1):
  9         stdout.write('*')
 10     print()
 11
 12 for i in range(3):
 13     for j in range(i+1):
 14         stdout.write(' ')
 15     for k in range(4 - 2 * i + 1):
 16         stdout.write('*')
 17     print() # 在python3中加上（）就能换行了
