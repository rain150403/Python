numpy 是使用python进行数据分析不可或缺的第三方库，非常多的科学计算工具都是基于 numpy 进行开发的。

ndarray对象是用于存放同类型元素的多维数组，是numpy中的基本对象之一，另一个是func对象。
本文主要内容是：
1 、简单介绍ndarray对象；
2、ndarray对象的常用属性；
3、如何创建ndarray对象；
4、ndarray元素访问。 
它的维度以及个维度上的元素个数由shape决定。

1 numpy.ndarray()

标题中的函数就是numpy的构造函数，我们可以使用这个函数创建一个ndarray对象。构造函数有如下几个可选参数：

参数		类型			作用
shape		int型tuple            多维数组的形状
dtype		data-type             数组中元素的类型
buffer					 用于初始化数组的buffer
offset		int			 buffer中用于初始化数组的首个数据的偏移
strides	int型tuple	       每个轴的下标增加1时，数据指针在内存中增加的字节数
order		‘C’ 或者 ‘F’		‘C’:行优先；’F’:列优先


实例：

>>> np.ndarray(shape=(2,3), dtype=int, buffer=np.array([1,2,3,4,5,6,7]), offset=0, order="C") 
array([[1, 2, 3],
       [4, 5, 6]])
>>> np.ndarray(shape=(2,3), dtype=int, buffer=np.array([1,2,3,4,5,6,7]), offset=0, order="F")
array([[1, 3, 5],
       [2, 4, 6]])
>>> np.ndarray(shape=(2,3), dtype=int, buffer=np.array([1,2,3,4,5,6,7]), offset=8, order="C") 
array([[2, 3, 4],
       [5, 6, 7]])


2 ndarray对象的常用属性

接下来介绍ndarray对象最常用的属性

属性			含义
T			转置，与self.transpose( )相同，如果维度小于2返回self
size		数组中元素个数
itemsize	数组中单个元素的字节长度
dtype		数组元素的数据类型对象
ndim		数组的维度
shape		数组的形状
data		指向存放数组数据的python buffer对象
flat		返回数组的一维迭代器
imag		返回数组的虚部
real		返回数组的实部
nbytes		数组中所有元素的字节长度


实例：

>>> a = np.array(range(15)).reshape(3,5)
>>> a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
>>> a.T
array([[ 0,  5, 10],
       [ 1,  6, 11],
       [ 2,  7, 12],
       [ 3,  8, 13],
       [ 4,  9, 14]])
>>> a.size
15
>>> a.itemsize
8
>>> a.ndim
2
>>> a.shape
(3, 5)
>>> a.dtype
dtype('int64')


3 创建ndarray

3.1 array

使用array函数，从常规的python列表或者元组中创建数组，元素的类型由原序列中的元素类型确定。

numpy.array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0)

实例：

>>> np.array([1, 2, 3])
array([1, 2, 3])
>>> np.array([[1, 2],[3, 4]])
array([[1, 2],
       [3, 4]])
>>> c = array( [ [1,2], [3,4] ], dtype=complex )
>>> c
array([[1.+0.j, 2.+0.j], 
       [3.+0.j, 4.+0.j]])
>>> a = np.array([1, 2, 3], ndmin=2)
>>> a
array([[1, 2, 3]])
>>> a.shape
(1, 3)
>>> np.array(np.mat('1 2; 3 4'))
array([[1, 2],
       [3, 4]])
>>> np.array(np.mat('1 2; 3 4'), subok=True)
matrix([[1, 2],
        [3, 4]])

subok为True，并且object是ndarray子类时（比如矩阵类型），返回的数组保留子类类型

3.2 ones与zeros系列函数

某些时候，我们在创建数组之前已经确定了数组的维度以及各维度的长度。这时我们就可以使用numpy内建的一些函数来创建ndarray。 
例如：函数ones创建一个全1的数组、函数zeros创建一个全0的数组、函数empty创建一个内容随机的数组，在默认情况下，用这些函数创建的数组的类型都是float64，若需要指定数据类型，只需要闲置dtype参数即可：

>>> a = np.ones(shape = (2, 3))    #可以通过元组指定数组形状
>>> a
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
>>> a.dtype
dtype('float64')
>>> b = np.zeros(shape = [3, 2], dtype=np.int64)    #也可以通过列表来指定数组形状，同时这里指定了数组类型
>>> b
array([[0, 0],
       [0, 0],
       [0, 0]])
>>> b.dtype
dtype('int64')
>>> c = np.empty((4,2))
>>> c
array([[  0.00000000e+000,   0.00000000e+000],
       [  6.92806325e-310,   6.92806326e-310],
       [  6.92806326e-310,   6.92806326e-310],
       [  0.00000000e+000,   0.00000000e+000]])

上述三个函数还有三个从已知的数组中，创建shape相同的多维数组：ones_like、zeros_like、empty_like，用法如下：

>>> a = [[1,2,3], [3,4,5]]
>>> b = np.zeros_like(a)
>>> b
array([[0, 0, 0],
       [0, 0, 0]])
#其他两个函数用法类似

除了上述几个用于创建数组的函数，还有如下几个特殊的函数：

函数名		用途
eye		生成对角线全1，其余位置全是0的二维数组
identity	生成单位矩阵
full		生成由固定值填充的数组
full_like	生成由固定值填充的、形状与给定数组相同的数组
特别地，eye函数的全1的对角线位置有参数k确定 
用法如下：

>>> np.eye(3, k = 0)    #k=0时，全1对角线为主对角线
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
>>> np.eye(3, k = 1)  #k>0时，全1对角线向上移动相应的位置
array([[ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 0.,  0.,  0.]])
>>> np.eye(3, k = -1)  #k<0时，全1对角线向下移动相应的位置
array([[ 0.,  0.,  0.],
       [ 1.,  0.,  0.],
       [ 0.,  1.,  0.]])
>>> np.identity(4)
array([[ 1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.]])
>>> np.full(shape = (2,2), fill_value = 2)
array([[ 2.,  2.],
       [ 2.,  2.]])
>>> np.full_like([[1,2,3],[3,4,5]], 3)
array([[3, 3, 3],
       [3, 3, 3]])

3.3 arange、linspace与logspace

arange函数类似python中的range函数，通过指定初始值、终值以及步长（默认步长为1）来创建数组
linspace函数通过指定初始值、终值以及元素个数来创建一维数组
logspace函数与linspace类似，只不过它创建的是一个等比数列，同样的也是一个一维数组 
实例：
>>> np.arange(0,10,2) 
array([0, 2, 4, 6, 8])
>>> np.arange(0,10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.linspace(0,10, 20)
array([  0.        ,   0.52631579,   1.05263158,   1.57894737,
         2.10526316,   2.63157895,   3.15789474,   3.68421053,
         4.21052632,   4.73684211,   5.26315789,   5.78947368,
         6.31578947,   6.84210526,   7.36842105,   7.89473684,
         8.42105263,   8.94736842,   9.47368421,  10.        ])
>>> np.logspace(0, 10, 10)
array([  1.00000000e+00,   1.29154967e+01,   1.66810054e+02,
         2.15443469e+03,   2.78255940e+04,   3.59381366e+05,
         4.64158883e+06,   5.99484250e+07,   7.74263683e+08,
         1.00000000e+10])

3.4 fromstring与fromfunction

fromstring函数从字符串中读取数据并创建数组
fromfunction函数由第一个参数作为计算每个数组元素的函数（函数对象或者lambda表达式均可），第二个参数为数组的形状 
实例：
>>> s1 = "1,2,3,4,5"
>>> np.fromstring(s1, dtype=np.int64, sep=",")
array([1, 2, 3, 4, 5])
>>> s2 = "1.01 2.23 3.53 4.76"
>>> np.fromstring(s2, dtype=np.float64, sep=" ")
array([ 1.01,  2.23,  3.53,  4.76])
>>> def func(i, j):
...     return (i+1)*(j+1)
... 
>>> np.fromfunction(func, (9,9))
array([[  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.],
       [  2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.],
       [  3.,   6.,   9.,  12.,  15.,  18.,  21.,  24.,  27.],
       [  4.,   8.,  12.,  16.,  20.,  24.,  28.,  32.,  36.],
       [  5.,  10.,  15.,  20.,  25.,  30.,  35.,  40.,  45.],
       [  6.,  12.,  18.,  24.,  30.,  36.,  42.,  48.,  54.],
       [  7.,  14.,  21.,  28.,  35.,  42.,  49.,  56.,  63.],
       [  8.,  16.,  24.,  32.,  40.,  48.,  56.,  64.,  72.],
       [  9.,  18.,  27.,  36.,  45.,  54.,  63.,  72.,  81.]])
>>> np.fromfunction(lambda i,j: i+j, (3,3), dtype = int)
array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4]])

除了上面两个函数还有其他几个类似的从外部获取数据并创建ndarray，比如：frombuffer、fromfile、fromiter，还没用过，等用到了在详细记录

4 ndarray创建特殊的二维数组

ndarray提供了一些创建二维数组的特殊函数。numpy中matrix是对二维数组ndarray进行了封装之后的子类。
这里介绍的关于二维数组的创建，返回的依旧是一个ndarray对象，而不是matrix子类。
关于matrix的创建和操作，待后续笔记详细描述。为了表述方便，下面依旧使用矩阵这一次来表示创建的二维数组。 

1. diag函数返回一个矩阵的对角线元素、或者创建一个对角阵，对角线由参数k控制 
2. diagflat函数以输入作为对角线元素，创建一个矩阵，对角线由参数k控制 
3. tri函数生成一个矩阵，在某对角线以下元素全为1，其余全为0，对角线由参数k控制 
4. tril函数输入一个矩阵，返回该矩阵的下三角矩阵，下三角的边界对角线由参数k控制 
5. triu函数与tril类似，返回的是矩阵的上三角矩阵 
6. vander函数输入一个一维数组，返回一个范德蒙德矩阵

#diag用法
>>> x = np.arange(9).reshape((3,3))
>>> x
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
>>> np.diag(x)
array([0, 4, 8])
>>> np.diag(x, k=1)
array([1, 5])
>>> np.diag(x, k=-1)
array([3, 7])
>>> np.diag(np.diag(x))
array([[0, 0, 0],
       [0, 4, 0],
       [0, 0, 8]])
>>> np.diag(np.diag(x), k=1)
array([[0, 0, 0, 0],
       [0, 0, 4, 0],
       [0, 0, 0, 8],
       [0, 0, 0, 0]])
#diagflat用法
>>> np.diagflat([[1,2],[3,4]])
array([[1, 0, 0, 0],
       [0, 2, 0, 0],
       [0, 0, 3, 0],
       [0, 0, 0, 4]])
>>> np.diagflat([1,2,3], k=-1)
array([[0, 0, 0, 0],
       [1, 0, 0, 0],
       [0, 2, 0, 0],
       [0, 0, 3, 0]])
#tri
>>> np.tri(3,4, k=1, dtype=int)  
array([[1, 1, 0, 0],
       [1, 1, 1, 0],
       [1, 1, 1, 1]])
>>> np.tri(3,4)
array([[ 1.,  0.,  0.,  0.],
       [ 1.,  1.,  0.,  0.],
       [ 1.,  1.,  1.,  0.]])
#tril与triu
>>> x = np.arange(12).reshape((3,4))
>>> x
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> np.tril(x, k=1) 
array([[ 0,  1,  0,  0],
       [ 4,  5,  6,  0],
       [ 8,  9, 10, 11]])
>>> np.triu(x, k=1) 
array([[ 0,  1,  2,  3],
       [ 0,  0,  6,  7],
       [ 0,  0,  0, 11]])
#vander
>>> np.vander([2,3,4,5])
array([[  8,   4,   2,   1],
       [ 27,   9,   3,   1],
       [ 64,  16,   4,   1],
       [125,  25,   5,   1]])
>>> np.vander([2,3,4,5], N=3)
array([[ 4,  2,  1],
       [ 9,  3,  1],
       [16,  4,  1],
       [25,  5,  1]])

5 ndarray元素访问

5.1 一维数组

对于一维的ndarray可以使用python访问内置list的方式进行访问：整数索引、切片、迭代等方式 
关于ndarray切片 
与内置list切片类似，形式： 
array[beg:end:slice] 
beg: 开始索引 
end: 结束索引（不包含这个元素） 
step: 间隔 
需要注意的是： 
1. beg可以为空，表示从索引0开始； 
2. end也可以为空，表示达到索引结束（包含最后一个元素）； 
3. step为空，表示间隔为1； 
4. 负值索引：倒数第一个元素的索引为-1，向前以此减1 
5. 负值step：从后往前获取元素

>>> x = np.arange(16)*4
>>> x
array([ 0,  4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60])
>>> x[11]
44
>>> x[4:9]
array([16, 20, 24, 28, 32])
>>> x[:10:3]
array([ 0, 12, 24, 36])
>>> x[0:13:2]
array([ 0,  8, 16, 24, 32, 40, 48])
>>> x[::-1]    #逆置数组
array([60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12,  8,  4,  0])
>>> print [val for val in x]    #迭代元素
[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]


特别注意的是，ndarray中的切片返回的数组中的元素是原数组元素的索引，对返回数组元素进行修改会影响原数组的值

>>> x[:-1]
array([ 0,  5, 10, 15, 20, 25, 30, 35, 40])
>>> y = x[::-1]
>>> y
array([45, 40, 35, 30, 25, 20, 15, 10,  5,  0])
>>> y[0] = 100    #修改y的首个元素的值
>>> y
array([100,  40,  35,  30,  25,  20,  15,  10,   5,   0])
>>> x      #x[-1]也被修改（本质上是一个元素）
array([  0,   5,  10,  15,  20,  25,  30,  35,  40, 100])

除了上述与list相似的访问元素的方式，ndarray有一种通过列表来指定要从ndarray中获取元素的索引，例如：

>>> x = np.arange(10)*5
>>> x
array([ 0,  5, 10, 15, 20, 25, 30, 35, 40, 45])
>>> x[[0, 2, 4, 5, 9]]    #指定获取索引为0、2、4、5、9的元素
array([ 0, 10, 20, 25, 45])

5.2 多维数组

多维ndarray中，每一维都叫一个轴axis。在ndarray中轴axis是非常重要的，有很多对于ndarray对象的运算都是基于axis进行，比如sum、mean等都会有一个axis参数（针对对这个轴axis进行某些运算操作），后续将会详细介绍。 
对于多维数组，因为每一个轴都有一个索引，所以这些索引由逗号进行分割，例如：

>>> x = np.arange(0, 100, 5).reshape(4, 5)
>>> x
array([[ 0,  5, 10, 15, 20],
       [25, 30, 35, 40, 45],
       [50, 55, 60, 65, 70],
       [75, 80, 85, 90, 95]])
>>> x[1,2]      #第1行，第2列
35
>>> x[1:4, 3]    #第1行到第3行中所有第3列的元素
array([40, 65, 90])
>>> x[:, 4]      #所有行中的所有第4列的元素
array([20, 45, 70, 95])
>>> x[0:3, :]    #第0行到第三行中所有列的元素
array([[ 0,  5, 10, 15, 20],
       [25, 30, 35, 40, 45],
       [50, 55, 60, 65, 70]])

需要注意的是： 
1. 当提供的索引比轴数少时，缺失的索引表示整个切片（只能缺失后边的轴） 
2. 当提供的索引为:时，也表示整个切片 
3. 可以使用...代替几个连续的:索引

>>> x[1:3]    #缺失第二个轴
array([[25, 30, 35, 40, 45],
       [50, 55, 60, 65, 70]])
>>> x[:, 0:4]      #第一个轴是 :
array([[ 0,  5, 10, 15],
       [25, 30, 35, 40],
       [50, 55, 60, 65],
       [75, 80, 85, 90]])
>>> x[..., 0:4]    #...代表了第一个轴的 : 索引
array([[ 0,  5, 10, 15],
       [25, 30, 35, 40],
       [50, 55, 60, 65],
       [75, 80, 85, 90]])

多维数组的迭代 
可以使用ndarray的flat属性迭代数组中每一个元素

>>> for item in x.flat:
...     print item,
...
0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95

>>> x = np.arange(0, 100, 5).reshape(4, 5)
>>> x
array([[ 0,  5, 10, 15, 20],
       [25, 30, 35, 40, 45],
       [50, 55, 60, 65, 70],
       [75, 80, 85, 90, 95]])
>>> for item in x.flat:
...     print item,
...
0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95
>>> for item in x.flat:
...     print item
...
0
5
10
15
20
25
30
35
40
45
50
55
60
65
70
75
80
85
90
95
>>>
看出输出的差别了吗？有逗号，横着输出，没逗号竖着输出。


#######################################
上一篇主要介绍了ndarray对象的一些基本属性以及创建ndarray对象的一些非常常用的方法。
接下来主要介绍ndarray对象比较常用的对象方法。
需要注意的是，以下ndarray对象方法也是numpy中的函数：all, any, argmax, argmin, argpartition, argsort, choose, clip, 
compress, copy, cumprod, cumsum, diagonal, imag, max, mean, min, nonzero, partition, prod, ptp, put, ravel, real, 
repeat, reshape, round, searchsorted, sort, squeeze, std, sum, swapaxes, take, trace, transpose, var。

1 数组转换方法

常用方法						功能
ndarray.item(*args)			复制数组中的一个元素，并返回
ndarray.tolist()			将数组转换成python标准list
ndarray.itemset(*args)		修改数组中某个元素的值
ndarray.tostring([order])	       构建一个包含ndarray的原始字节数据的字节字符串
ndarray.tobytes([order])	       功能同tostring
ndarray.byteswap(inplace)	       将ndarray中每个元素中的字节进行大小端转换
ndarray.copy([order])		复制数组并返回（深拷贝）
ndarray.fill(value)			使用值value填充数组

示例：

>>> a = np.random.randint(12, size=(3,4)) # 随机产生0-11的12个整数，可重复
>>> a
array([[11,  1,  0, 11],
       [11,  0,  4,  6],
       [ 0,  1,  6,  7]])
>>> a.item(7)      #获取第7个元素
6
>>> a.item((1, 2))    #获取元组对应的元素
4
>>> a.itemset(7, 111)    #设置元素
>>> a
array([[ 11,   1,   0,  11],
       [ 11,   0,   4, 111],
       [  0,   1,   6,   7]])
>>> a.itemset((1, 2), 12)
>>> a
array([[ 11,   1,   0,  11],
       [ 11,   0,  12, 111],
       [  0,   1,   6,   7]])
>>> a.tolist()      #返回python标准列表，
[[11, 1, 0, 11], [11, 0, 4, 6], [0, 1, 6, 7]]
>>> a.tostring()
'\x0b\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00'
>>> b = a.copy()      #深拷贝，b与a是两个无关的数组
>>> b[0,0] = 10
>>> b
array([[10,  1,  0, 11],
       [11,  0,  4,  6],
       [ 0,  1,  6,  7]])
>>> a
array([[11,  1,  0, 11],
       [11,  0,  4,  6],
       [ 0,  1,  6,  7]])
>>> b.fill(9)
>>> b
array([[9, 9, 9, 9],
       [9, 9, 9, 9],
       [9, 9, 9, 9]])
>>> a.byteswap()      #元素大小端转换，a不变
array([[ 792633534417207296,   72057594037927936,                   0,
         792633534417207296],
       [ 792633534417207296,                   0,  864691128455135232,
        7998392938210000896],
       [                  0,   72057594037927936,  432345564227567616,
         504403158265495552]])
>>> a
array([[ 11,   1,   0,  11],
       [ 11,   0,  12, 111],
       [  0,   1,   6,   7]])
>>> a.byteswap(True)    #原地转换，a被修改
array([[ 792633534417207296,   72057594037927936,                   0,
         792633534417207296],
       [ 792633534417207296,                   0,  864691128455135232,
        7998392938210000896],
       [                  0,   72057594037927936,  432345564227567616,
         504403158265495552]])
>>> a
array([[ 792633534417207296,   72057594037927936,                   0,
         792633534417207296],
       [ 792633534417207296,                   0,  864691128455135232,
        7998392938210000896],
       [                  0,   72057594037927936,  432345564227567616,
         504403158265495552]])

如下几个方法：ndarray.tofile, ndarray.dump, ndarray.dumps, ndarray.astype, ndarray.view, ndarray.getfield, ndarray.setflags,
还没有用过，暂时不对其进行详细介绍，等用到了在补充。关于这些方法的详细介绍可以查阅numpy的官方文档。

2 形状操作

常用方法									功能
ndarray.reshape(shape[,order])			返回一个具有相同数据域，但shape不一样的视图
ndarray.resize(new_shape[,orefcheck])	       原地修改数组的形状（需要保持元素个数前后相同）
ndarray.transpose(*axes)				返回数组针对某一轴进行转置的视图
ndarray.swapaxes(axis1, asix2)			返回数组axis1轴与axis2轴互换的视图
ndarray.flatten([order])				返回将原数组压缩成一维数组的拷贝（全新的数组）
ndarray.ravel([order])				返回将原数组压缩成一维数组的视图
ndarray.squeeze([axis])				返回将原数组中的shape中axis==1的轴移除之后的视图

注意事项！！！ 
上述方法中，除resize、flatten外其他的方法返回的都是原数组修改shape或者axes之后的视图，也就是说，
对返回数组中的元素进行修改，原数组中对应的元素也会被修改（因为它们是公用同一个数据域的）。同时，
resize方法会修改原数组的shape属性，其他方法不会修改原数组任何内部数据。 
示例：

>>> x = np.arange(0,12)
>>> x
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
>>> x_reshape = x.reshape((3,4))
>>> x_reshape
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> x_reshape[0,0] = 10    #修改返回数组的元素，直接影响原数组中的元素值（数据域是相同的）
>>> x
array([10,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
>>> x[0,0] = 0 

>>> x.resize((3,4))    #resize没有返回值，会直接修改数组的shape，如下所示
>>> x
array([[10,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

>>> x_transpose = x.transpose()  #对于二维数组，返回数组的转置
>>> x_transpose
array([[ 0,  4,  8],
       [ 1,  5,  9],
       [ 2,  6, 10],
       [ 3,  7, 11]])
>>> x.resize(2,2,3)
>>> x
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]]])

>>> x.swapaxes(0,2)  #本质上还是修改strides以及shape
array([[[ 0,  6],
        [ 3,  9]],
       [[ 1,  7],
        [ 4, 10]],
       [[ 2,  8],
        [ 5, 11]]])
>>> x.swapaxes(0,2).strides  #互换strides中第0、2位置上的数值
(8, 24, 48)
>>> x.strides
(48, 24, 8)

>>> x
array([[[ 0,  1,  2],
        [ 3,  4,  5]],
       [[ 6,  7,  8],
        [ 9, 10, 11]]])
>>> y = x.flatten()  #返回一个全新的数组
>>> y
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
>>> y[0] = 100
>>> y
array([100,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11])
>>> x  #修改y中元素的值，不影响x中的元素
array([[[ 0,  1,  2],
        [ 3,  4,  5]],
       [[ 6,  7,  8],
        [ 9, 10, 11]]])

>>> x_ravel = x.ravel()  #与flatten类似，但是返回的是原数组的视图，数据域是相同的
>>> x_ravel
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

>>> x.resize((1,2,2,3,1))
>>> x
array([[[[[ 0],
          [ 1],
          [ 2]],
         [[ 3],
          [ 4],
          [ 5]]],
        [[[ 6],
          [ 7],
          [ 8]],
         [[ 9],
          [10],
          [11]]]]])
>>> x.squeeze()  #移除shape中值为1的项
array([[[ 0,  1,  2],
        [ 3,  4,  5]],
       [[ 6,  7,  8],
        [ 9, 10, 11]]])
>>> x.shape
(1, 2, 2, 3, 1)
>>> x.squeeze().shape
(2, 2, 3)


tips： transpose 
1. transpose的本质是按照参数axes修改了strides以及shape属性（自己的理解）： 
1）提供axes，按照axes中提供的各个轴的新位置调整strides属性中对应位置上的值 
2）不提供axes，取原strides的对称形式作为返回数组的strides属性 
3）shape属性也是按照上述方式修改 
示例：

>>> x  #先看二维的情形
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> x.transpose()  #不提供axes，等同于x.transpose((1, 0))
array([[ 0,  4,  8],
       [ 1,  5,  9],
       [ 2,  6, 10],
       [ 3,  7, 11]])
>>> x.strides   # 搞不懂strides到底是怎么回事？？？
(32, 8)
>>> x.transpose().strides
(8, 32)
>>> x.transpose((0,1))  #提供axes，各个轴的位置没有变化，结果与x是一样的
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> x.transpose((0,1)).strides  #strides参数也没有变化
(32, 8)
>>> y  #再看三维的情形
array([[[ 0,  1,  2],
        [ 3,  4,  5]],
       [[ 6,  7,  8],
        [ 9, 10, 11]]])
>>> y.transpose()  #不带axes参数，等同于y.transpose((2,1,0))
array([[[ 0,  6],
        [ 3,  9]],
       [[ 1,  7],
        [ 4, 10]],
       [[ 2,  8],
        [ 5, 11]]])
>>> y.transpose().strides
(8, 24, 48)
>>> y.strides
(48, 24, 8)
>>> y.transpose((2,0,1))  #带axes参数
array([[[ 0,  3],
        [ 6,  9]],
       [[ 1,  4],
        [ 7, 10]],
       [[ 2,  5],
        [ 8, 11]]])
>>> y.transpose((2,0,1)).strides  #注意strides各个参数的位置与原来strides参数的位置，正好是转置时axes的对应位置
(8, 48, 24)


不提供axes参数时，转换前后shape、strides以及数据有如下关系： 
1）转置前a.shape=(i[0], i[1], ..., i[n-2], i[n-1])，转置后的：a.transpose().shape=(i[n-1], i[n-2], ..., i[1], i[0]) 
2）转置前`a.strides=(j[0], j[1], ..., j[n-2], j[n-1])，转置后：a.transpose().strides=(j[n-1], j[n-2], ..., j[1], j[0]) 
3）数据：a[i[0], i[1], ..., i[n-2], i[n-1]] == a.transpose()[i[n-1], i[n-2], ..., i[1], i[0]]
思考提供axes参数时，上述关系是怎样的？

3 计算

关于ndarray对象的很多计算方法都有一个axis参数，它有如下作用： 
1. 当axis=None（默认）时，数组被当成一个一维数组，对数组的计算操作是对整个数组进行的，比如sum方法，就是求数组中所有元素的和； 
2. 当axis被指定为一个int整数时，对数组的计算操作是以提供的axis轴进行的。 
示例：

>>> a
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
>>> a.sum(0)  #等价于a[0,:,:] + a[1,:,:]
array([[12, 14, 16, 18],
       [20, 22, 24, 26],
       [28, 30, 32, 34]])
>>> a.sum(1)  #等价于a[:,0,:] + a[:,1,:] + a[:,2,:]
array([[12, 15, 18, 21],
       [48, 51, 54, 57]])
>>> a.sum(2)  #等价于a[:,:,0]+a[:,:,1]+a[:,:,2]+a[:,:,3]
array([[ 6, 22, 38],
       [54, 70, 86]])

常用方法								功能
ndarray.max([axis, out, keepdims])				返回根据指定的axis计算最大值
ndarray.argmax([axis, out])					返回根据指定axis计算最大值的索引
ndarray.min([axis, out, keepdims])				返回根据指定的axis计算最小值
ndarray.argmin([axis, out])					返回根据指定axis计算最小值的索引
ndarray.ptp([axis, out])					返回根据指定axis计算最大值与最小值的差
ndarray.clip([min, max, out])				返回数组元素限制在[min, max]之间的新数组（小于min的转为min，大于max的转为max）
ndarray.round([decimals, out])				返回指定精度的数组（四舍五入）
ndarray.trace([offset, axis1, axis2, dtype, out])	返回数组的迹（对角线元素的和）
ndarray.sum([axis, dtype, out, keepdims])			根据指定axis计算数组的和，默认求所有元素的和
ndarray.cumsum([axis, dtype, out])				根据指定axis计算数组的累积和
ndarray.mean([axis, dtype, out, keepdims])		根据指定axis计算数组的平均值
ndarray.var([axis, dtype, out, ddof, keepdims])		根据指定的axis计算数组的方差
ndarray.std([axis, dtype, out, ddof, keepdims])		根据指定axis计算数组的标准差
ndarray.prod([axis, dtype, out, keepdims])		根据指定axis计算数组的积
ndarray.cumprod([axis, dtype, out])			根据指定axis计算数据的累计积
ndarray.all([axis, dtype, out])				根据指定axis判断所有元素是否全部为真
ndarray.any([axis, out, keepdims])				根据指定axis判断是否有元素为真

>>> a
array([[2, 3, 4, 9],
       [8, 7, 6, 5],
       [4, 3, 5, 8]])
>>> a.max()
9
>>> a.max(axis=0)  #shape=(4,)，即原shape去掉第0个axis
array([8, 7, 6, 9])
>>> a.max(axis=1)  #shape=(3,)，即原shape去掉第1个axis
array([9, 8, 8])
>>> a.argmax()
3
>>> a.argmax(axis=0)
array([1, 1, 1, 0])
>>> a.argmax(axis=1)
array([3, 0, 3])
>>> b = a.flatten()
>>> b
array([2, 3, 4, 9, 8, 7, 6, 5, 4, 3, 5, 8])
>>> b.clip(3, 5)  #数组元素限定在[3, 5]之间
array([3, 3, 4, 5, 5, 5, 5, 5, 4, 3, 5, 5])
>>> a.resize((2,2,3))
>>> a
array([[[2, 3, 4],
        [9, 8, 7]],
       [[6, 5, 4],
        [3, 5, 8]]])
>>> a.trace(0,axis=0, axis=1)  #等同于[trace(a[:,:,0]), trace(a[:,:,1], trace(a[:,:,2])]；shape=(3,)，即原shape去掉第0，1个axis
array([ 5,  8, 12])
>>> np.eye(3).trace()  #对角线元素的和
3.0
>>> b.reshape((3,4))
array([[2, 3, 4, 9],
       [8, 7, 6, 5],
       [4, 3, 5, 8]])
>>> b.reshape((3,4)).std(0)
array([ 2.49443826,  1.88561808,  0.81649658,  1.69967317])
>>> b.reshape((3,4)).std(1)
array([ 2.6925824 ,  1.11803399,  1.87082869])
>>> x = np.arange(8) 
>>> x
array([0, 1, 2, 3, 4, 5, 6, 7])
>>> x.cumsum()  #累计和
array([ 0,  1,  3,  6, 10, 15, 21, 28])
>>> x = np.arange(1,9)
>>> x
array([1, 2, 3, 4, 5, 6, 7, 8])
>>> x.cumprod()  #累计积
array([    1,     2,     6,    24,   120,   720,  5040, 40320])


4 选择元素以及操作

常用方法									方法功能
ndarray.take(indices[, axis, out, model])			从原数组中根据指定的索引获取对应元素，并构成一个新的数组返回
ndarray.put(indices, values[, mode])			将数组中indices指定的位置设置为values中对应的元素值
ndarray.repeat(repeats[, axis])				根据指定的axis重复数组中的元素
ndarray.sort([axis, kind, order])				原地对数组元素进行排序
ndarray.argsort([axis, kind, order])			返回对数组进行升序排序之后的索引
ndarray.partition(kth[, axis, kind, order])		将数组重新排列，所有小于kth的值在kth的左侧，所有大于或等于kth的值在kth的右侧
ndarray.argpartition(kth[, axis, kind, order])		对数组执行partition之后的元素索引
ndarray.searchsorted(v[, side, sorter])			若将v插入到当前有序的数组中，返回插入的位置索引
ndarray.nonzero()						返回数组中非零元素的索引
ndarray.diagonal([offset, axis1, axis2])			返回指定的对角线

>>> a
array([2, 3, 4, 9, 8, 7, 6, 5, 4, 3, 5, 8])
>>> a.take([0,3,6])
array([2, 9, 6])
>>> a.take([[2, 5], [3,6]])  #返回数组的形状与indices形状相同
array([[4, 7],
       [9, 6]])
>>> a.put([0, -1], [0, 111])
>>> a
array([  0,   3,   4,   9,   8,   7,   6,   5,   4,   3,   5, 111])

>>> a.sort()  #原地排序
>>> a
array([  0,   3,   3,   4,   4,   5,   5,   6,   7,   8,   9, 111])

>>> b
array([  0,   3,   4,   9,   8,   7,   6,   5,   4,   3,   5, 111])
>>> b_idx = b.argsort()  #获取排序索引
>>> b_idx
array([ 0,  1,  9,  2,  8,  7, 10,  6,  5,  4,  3, 11])
>>> b[b_idx]  #根据排序索引获取b中元素，正好是排序好的
array([  0,   3,   3,   4,   4,   5,   5,   6,   7,   8,   9, 111])

>>> c
array([  0,   3,   4,   9,   8,   7,   6,   5,   4,   3,   5, 111])
>>> c.partition(5)
>>> c
array([  3,   4,   4,   0,   3,   5,   6,   5,   7,   8,   9, 111])

>>> a
array([  0,   3,   3,   4,   4,   5,   5,   6,   7,   8,   9, 111])
>>> a.searchsorted(2)
1
>>> a.searchsorted(10)
11
>>> a.searchsorted(3)  #side默认为lift
1
>>> a.searchsorted(3, side="right") 
3

>>> e = np.eye(4)
>>> e
array([[ 1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.]])
>>> e.nonzero()
(array([0, 1, 2, 3]), array([0, 1, 2, 3]))
>>> e[e.nonzero()]
array([ 1.,  1.,  1.,  1.])

>>> a = np.arange(4).reshape(2,2)
>>> a
array([[0, 1],
       [2, 3]])
>>> a.diagonal()  #二维数组取对角线元素
array([0, 3])
>>> a = np.arange(8).reshape(2,2,2)
>>> a
array([[[0, 1],
        [2, 3]],
       [[4, 5],
        [6, 7]]])
>>> a.diagonal(offset=0, axis1=0, axis2=1)  #三维数组根据指定axis取对角线，本质是取下面两个二维数组的对角线
array([[0, 6],
       [1, 7]])
>>> a[:,:,0]    #对角线[0, 6]
array([[0, 2],
       [4, 6]])
>>> a[:,:,1]    #对角线[1, 7]
array([[1, 3],
       [5, 7]])
>>> a.diagonal(offset=0, axis1=0, axis2=2)
array([[0, 5],
       [2, 7]])
>>> a[:,0,:]
array([[0, 1],
       [4, 5]])
>>> a[:,1,:]
array([[2, 3],
       [6, 7]])
>>> a.diagonal(offset=0, axis1=1, axis2=2) 
array([[0, 3],
       [4, 7]])
>>> a[0,:,:]
array([[0, 1],
       [2, 3]])
>>> a[1,:,:]
array([[4, 5],
       [6, 7]])

##################################################
numpy的基本属性：

NumPy的主要对象是同种元素的多维数组。这是一个所有的元素都是一种类型、通过一个正整数元组索引的元素表格(通常是元素是数字)。
在NumPy中维度(dimensions)叫做轴(axes)，轴的个数叫做秩(rank)。

     Numpy中提供的核心对象 array

NumPy的数组类被称作 ndarray 。通常被称作数组。注意 numpy.array 和标准Python库类 array.array 并不相同，后者只处理一维数组和提供少量功能。
更多重要ndarray对象属性有：

1. ndarray.ndim

数组轴的个数，在python的世界中，轴的个数被称作秩

2. ndarray.shape

数组的维度。这是一个指示数组在每个维度上大小的整数元组。例如一个n排m列的矩阵，它的shape属性将是(2,3),这个元组的长度显然是秩，即维度或者ndim属性

3. ndarray.size

数组元素的总个数，等于shape属性中元组元素的乘积。

4. ndarray.dtype

一个用来描述数组中元素类型的对象，可以通过创造或指定dtype使用标准Python类型。另外NumPy提供它自己的数据类型。

     提供的dtype类型：  
bool_	Boolean (True or False) stored as a byte
int_	Default integer type (same as C long; normally either int64 or int32)
intc	Identical to C int (normally int32 or int64)
intp	Integer used for indexing (same as C ssize_t; normally either int32 or int64)
int8	Byte (-128 to 127)
int16	Integer (-32768 to 32767)
int32	Integer (-2147483648 to 2147483647)
int64	Integer (-9223372036854775808 to 9223372036854775807)
uint8	Unsigned integer (0 to 255)
uint16	Unsigned integer (0 to 65535)
uint32	Unsigned integer (0 to 4294967295)
uint64	Unsigned integer (0 to 18446744073709551615)
float_	Shorthand for float64.
float16	Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
float32	Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
float64	Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
complex_	Shorthand for complex128.
complex64	Complex number, represented by two 32-bit floats (real and imaginary components)
complex128	Complex number, represented by two 64-bit floats (real and imaginary components)
ndarray.itemsize

数组中每个元素的字节大小。例如，一个元素类型为float64的数组itemsiz属性值为8(=64/8),又如，一个元素类型为complex32的数组item属性为4(=32/8).

5. ndarray.data

包含实际数组元素的缓冲区，通常我们不需要使用这个属性，因为我们总是通过索引来使用数组中的元素。



################################################################################################
这一节简要介绍ndarray和func————

numpy基础——from——《利用Python进行数据分析》

Numpy简介

Numpy（Numerical Python的简称）是高性能科学计算和数据分析的基础包。其部分功能如下：

①ndarray，一个具有矢量算术运算和复杂广播能力的快速且节省空间的多维数组。

②用于对整组数据进行快速运算的标准数学函数（无需编写循环）。

③用于读写磁盘数据的工具以及用于操作内存映射文件的工具。

④线性代数、随机数生成以及傅里叶变换功能。

⑤用于集成由C、C++、Fortran等语言编写的代码的工具。

一、NumPy的ndarray：一种多维数组对象

每个数组都有一个shape（一个表示各维度大小的元组）和一个dtype（一个用于说明数组数据类型的对象）

1、创建ndarray

import numpy as np
data1 = [6,7.5,8,0,1]  #创建列表
arr1 = np.array(data1) #转换为数组
print(arr1) 
print(arr1.dtype) #数据类型保存在dtype对象中
data2 = [[1,2,3,4],[5,6,7,8]] #创建嵌套序列（由等长列表组成的列表）
arr2 = np.array(data2) #转换为多维数组
print(arr2)
print(arr1.dtype)
print(arr1.shape)

除np.array之外，还有一些函数也可以新建数组。比如，zeros和ones分别可以创建指定长度或形状的全0或全1数组。
empty可以创建一个没有任何具体值的数组。要用这些方法创建多维数组，只需传入一个表示形状的元组即可：

print(np.zeros(10)) #创建指定长度(10)的全0数组
print(np.ones((2,4)) #创建指定长度的(2行4列二维)的全1数组 ，要注意，这里是两个括号！！！
print(np.empty((2, 3, 2)))
 

第一个参数表示数组的维度

注意：认为np.empty会返回全0数组的想法是不安全的。很多情况下（如前所示），它返回的都是一些未初始化的垃圾值。***arange是Python内置函数range的数组版***：

range(10) #创建指定数量的顺序列表（内置函数,默认0开始）
print(np.arange(15)) #创建指定数量的顺序数组
arr1=np.array([1,2,3],dtype=np.float64)    #创建特定数据类型的数组
数组创建函数：



2、ndarray的数据类型
dtype（数据类型）是一个特殊的对象，它含有ndarray将一块内存解释为特定数据类型所需的信息。

dtype是NumPy如此强大和灵活的原因之一。多数情况下，它们直接映射到相应的机器表示，这使得“读写磁盘上的二进制数据流”
以及“集成低级语言代码（如C、Fortran）”等工作变得更加简单。
数值型dtype的命名方式相同：一个类型名（如float或int），后面跟一个用于表示各元素位长的数字。
标准的双精度浮点值（即Python中的float对象）需要占用8字节（即64位）。因此，该类型在NumPy中就记作float64。下图列示了NumPy所支持的全部数据类型。



注意：记不住这些NumPy的dtype也没关系，新手更是如此。通常只需要知道你所处理的数据的大致类型是浮点数、复数、整数、布尔值、字符串，还是普通的Python对象即可。
***当你需要控制数据在内存和磁盘中的存储方式时（尤其是对大数据集），那就得了解如何控制存储类型。***

转换数据类型

可以通过ndarray的astype方法显式地转换其dtype。

arr = np.array([1,2,3,4,5])
print(arr.dtype)
float_arr = arr.astype(np.float32)
print(float_arr.dtype)
注意：浮点数（比如float64和float32）只能表示近似的分数值。在复杂计算中，由于可能会积累一些浮点错误，因此比较操作只能在一定小数位以内有效。

3、数组和标量之间的运算

数组很重要，因为它使你不用编写循环即可对数据执行批量运算。这通常就叫做矢量化（vectorization）。***大小相等的数组之间的任何算术运算都会将运算应用到元素级。***

arr=np.array([[1.,2.,3.],[4.,5.,6.]])    #创建二维数组  
arr*arr    #行列号相同的数组元素间运算  
arr-arr  
1/arr  
arr*0.5  

4、基本的索引和切片

从表面上看，它们跟Python列表的功能差不多，跟列表最重要的区别在于，数组切片是原始数组的视图。这意味着数据不会被复制，视图上的任何修改都会直接反映到源数组上：

arr=np.arange(10)  
arr[5]    #索引第6个元素  
arr[5:8]    #索引第6到第9个元素作为数组  
arr[5:8]=12    #令第6到第9个元素等于12  
arr_slice=arr[5:8]    #数组切片是原始数据的视图，视图上的任何修改都会反映到原数组  
arr_slice[:]=64    #将数组切片的全部元素改为64  
arr[5:8].copy()    #得到数组切片的一份副本  
arr2d=np.array([[1,2,3],[4,5,6],[7,8,9]])  
arr2d[2]    #索引二维数组第3行  
arr2d[0][2]  arr2d[0,2]    #等价索引1行3列元素  
arr2d[:2]    #索引第1行和第2行（不含第3行）  
arr2d[:,:1]    #索引第1列  
arr2d[:-2]    #使用负数索引将从尾部开始选取行 

二维数组的索引方式

5、切片索引

ndarray的切片语法跟Python列表这样的一维对象差不多

可以一次性传入多个切片，就像传入多个索引那样。

6、布尔型索引

names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
names =='Bob' 
#array([ True, False, False,  True, False, False, False], dtype=bool)
data = np.random.randn(7,4)
data[names=='Bob']
data[names=='Bob',2:] #布尔型数组的长度必须跟被索引的轴长度一致
names !='Bob' #不等于
data[-(names=='Bob')] #负号(-)对条件进行否定
#Python关键字and和or在布尔型数组中无效。

通过布尔型数组设置值是一种经常用到的手段。为了将data中的所有负值都设置为0，我们可以这样：

data[data < 0] = 0

7、花式索引

指的是利用整数数组进行索引,以特定的顺序选取子行。

arr = np.empty((8,4))
for i in range(8):
    arr[i] = i
arr[[2,3,0]] #以特定顺序选取行子集，传入整数列表或ndarray
如果想要选出的是矩形区域，有两种方法，一种是data[[1,3]][:,[1,3]]，相当于先在轴0做一次索引，然后再在轴1做一次索引。
另一种是用np.ix_函数将两个一维数组转化成选取矩形的索引器:data[np.ix_([1,3],[1,3])]。

用np.ix_函数，它可以将两个一维整数数组转换为一个用于选取方形区域的索引器。

8、数组转置和轴对换

转置（transpose）是重塑的一种特殊形式，它返回的是源数据的视图（不会进行任何复制操作）。数组不仅有transpose方法，还有一个特殊的T属性。

 

二、通用函数：快速的元素级数组函数
通用函数（即ufunc）是一种对ndarray中的数据执行元素级运算的函数。你可以将其看做简单函数（接受一个或多个标量值，并产生一个或多个标量值）的矢量化包装器。



import numpy as np
arr=np.arange(10)  
np.sqrt(arr)    #计算各元素的平方根（arr**0.5）  
exp  #计算各元素指数ex;  abs  #绝对值;  
np.add(x,y)  #x、y数组中对应元素相加;  subtract #相减;  multiply #相乘;  divide #相除;  
1、利用数组进行数据处理

用数组表达式代替循环的做法，通常被称为矢量化。

import numpy as np
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points,points)
z = np.sqrt(xs **2 + ys **2)
print(z)
2、将条件逻辑表述为数组运算

numpy.where函数是三元表达式x if condition else y的矢量化版本。

如：

import numpy as np
xarr = np.array([1.1,1.2,1.3,1.4,1.5])
yarr = np.array([2.1,2.2,2.3,2.4,2.5])
cond = np.array([True,False,True,True,False])
假设我们想要根据cond中的值选取xarr和yarr的值：当cond中的值为True时，选取xarr的值，否则从yarr中选取。列表推导式的写法应该如下所示：

 

使用np.where

import numpy as np
xarr=np.array([1.1,1.2,1.3,1.4,1.5])    #两个数值数组  
yarr=np.array([2.1,2.2,2.3,2.4,2.5])  
cond=np.array([True,False,True,True,False])    #一个布尔数组  
result=np.where(cond,xarr,yarr)    #三元表达式  
3、数学和统计方法
可以通过数组上的一组数学函数对整个数组或某个轴向的数据进行统计计算。sum、mean以及标准差std等聚合计算（aggregation，通常叫做约简（reduction））既可以当做数组的实例方法调用，也可以当做顶级NumPy函数使用：

arr=np.random.randn(5,4)  
arr.mean();  np.mean(arr);  arr.sum();  
arr.mean(axis=1)    #计算该轴上的统计值（0为列，1为行）  
4、用于布尔型数组的方法

布尔值会被强制转换为1（True）和0（False）。因此，sum经常被用来对布尔型数组中的True值计数：

arr=randn(100)  
(arr>0).sum()    #正值的数量  
bools.any()    #用于测试数组中是否存在一个或多个True  
bools.all()    #用于测试数组中所有值是否都是True   
5、排序

跟Python内置的列表类型一样，NumPy数组也可以通过sort方法就地排序：

arr=randn(8)  
arr.sort()  
arr=randn(5,3)  
arr.sort(0)  #二维数组按列排序;  arr.sort(1)  #二维数组按行排序
6、唯一化

ints=np.array([3,3,3,2,2,1,1,4,4])  
np.unique(names)    #找出数组中的唯一值并返回已排序的结果


7、用于数组的文件输入输出

Numpy能够读写磁盘上的文本数据或二进制数据。

arr=np.arange(10)  
np.save('some_array',arr)  #数组以未压缩的原始二进制格式保存在.npy文件中  
np.load('some_array')  #通过np.load读取磁盘上的数组  
np.savez('array_archive.npz',a=arr,b=arr)  #将多个数组以保存在一个压缩文件中  
a=np.arange(0,12,0.5).reshape(4,-1)  
np.savetxt(r'E:\\knakan\\a.txt',a)  #缺省按照’%.18e’格式保存数据，以空格分隔  
np.loadtxt(r'E:\\kankan\\a.txt')
np.savetxt(r'\kankan\\a.txt',a,fmt="%d"delimiter=",")  #改为保存为整数，以逗号分隔  
np.loadtxt(r'E:\\kankan\\a.txt',delimiter=",")  #读入时也需指定逗号分隔  

8、 线性代数

x=np.array([[1.,2.,3.],[4.,5.,6.]])  
y=np.array([[6.,23.],[-1,7],[8,9]])  
x.dot(y)  #矩阵乘法，相当于np.dot(x,y)   
