# Softmax

Softmax 是逻辑回归的推广，用于解决多分类任务

![1581042298866](C:\Users\Liang\AppData\Roaming\Typora\typora-user-images\1581042298866.png)



对于逻辑回归，线性部分的预测结果为连续值，需要将这个连续值代入 Sigmoid 函数，转换为对应的条件概率，Softmax 是为了解决多分类问题而生的，同样要将预测结果代入到某个函数，而这个函数就是 Softmax 函数



#### 逻辑回归 -> 二分类

只需要学习一组参数 [ w1, w2, w3, w4...wd]
p(y=1|x)
p(y=0|x) = 1-p(y=1|x)

#### Softmax -> 多分类
p(y=1|x)  w

p(y=2|x)  w`

p(y=3|x)  w``
每个类别，对应要学习输入他们自己的一组参数

![1581043196950](C:\Users\Liang\AppData\Roaming\Typora\typora-user-images\1581043196950.png)

![1581173052249](C:\Users\Liang\AppData\Roaming\Typora\typora-user-images\1581173052249.png)

![1581173068792](C:\Users\Liang\AppData\Roaming\Typora\typora-user-images\1581173068792.png)

![1581173102228](C:\Users\Liang\AppData\Roaming\Typora\typora-user-images\1581173102228.png)

![1581173213814](C:\Users\Liang\AppData\Roaming\Typora\typora-user-images\1581173213814.png)

![1581173396339](C:\Users\Liang\AppData\Roaming\Typora\typora-user-images\1581173396339.png)