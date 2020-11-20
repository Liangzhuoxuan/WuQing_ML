[TOC]

## xgBoost推导


### 1. xgboost的目标函数

xgboost的目标函数定义为：损失函数 + 所有树的复杂度
$$
Obj = \sum^n_{i=1}l(y_i, \hat{y_i}) + \sum^K_{k=1}Ω(f_k)\tag{1}
$$
n为样本集的样本数，$l(y_i, \hat{y_i})$ 是定义的损失函数，根据样本的标签和预测值得到损失值，K表示基模型（树）的数量，$Ω(f_k)$表示第 k 颗树的复杂度，$f_k$表示第$k$颗树



其中对于样本 $i$ 的预测值为：将这个样本的特征矩阵 $x_i$ 带入整个模型的每一颗树里面的预测结果的加和
$$
\hat{y_i} = \sum^K_{k=1}f(x_i)\tag{2}
$$




### 2. 第t颗树的学习

对于第 t 次迭代
$$
\hat{y_i}^{(t)} = \hat{y_i}^{(t-1)} + f_t(x_i)\tag{3}
$$


将 (3) 代入 (1) 得：
$$
Obj^{(t)} = \sum^n_{i=1}l(y_i, \hat{y_i}^{(t-1)} + f_t(x_i)) + \sum^{t-1}_{k=1}Ω(f_k) + Ω(f_t)\tag{4}
$$
其中 $\hat{y_i}^{(t-1)}$ 和 $\sum^{t-1}_{k=1}Ω(f_k)$ 都是常量，只有一个变量，就是在第 t 次迭代的时候要求的 $f_t(x_i)$



### 3. 对目标函数进行二阶泰勒展开



为了简化目标函数 $Obj^{(t)}$ ，对其进行二阶泰勒展开：

二阶泰勒公式：
$$
f(x + \Delta{x}) \approx f(x) + f'(x)\Delta{x} + \frac{1}{2}f''(x)\Delta{x}
$$
其中，把 $\Delta{x}$ 当作 $f_t(x_i)$ ，$x$ 当作 $\hat{y_i}^{(t-1)}$


$$
Obj^{(t)} = \sum^n_{i=1}\Big{[} l(y_i, \hat{y_i}^{(t-1)} + \frac{\partial l(y_i, \hat{y_i}^{(t-1)})}{\partial \hat{y_i}^{(t-1)}}f_t(x_i) + \frac{1}{2}\frac{\partial^{2}l(y_i, \hat{y_i}^{(t-1)})}{\partial \hat{y_i}^{(t-1)}{^{2}}}      
f_t(x_i)^2\Big{]} + Ω(f_t) +\sum^{t-1}_{k=1}Ω(f_k) \tag{5}
$$


令损失函数 $l$ 关于 $\hat{y_i}^{(t-1)}$的一阶偏导和二阶偏导分别为 $g_i$  和 $h_i$ 
$$
g_i =\frac{\partial l(y_i, \hat{y_i}^{(t-1)})}{\partial \hat{y_i}^{(t-1)}}\tag{6}\\


h_i = \frac{\partial^{2}l(y_i, \hat{y_i}^{(t-1)})}{\partial \hat{y_i}^{(t-1)}{^{2}}}
$$


将 (6) 的 代入 (5) 并去掉里面的常数项，最后得到目标函数：
$$
Obj^{(t)} = \sum^n_{i=1}\Big{[}g_if_t(x_i) + \frac{1}{2}h_if_t(x_i)^2) \Big{]} + Ω(f_t)\tag{7}
$$


### 4. 定义一颗树

一颗树由以下两方面决定：

- 叶子结点的权重向量 $w_j$
- 样本到叶子结点的映射（树的分支结构）$q$

一个样本对应的预测值，就是把样本放到分给模型里面的每一颗树，样本所落在的叶子结点的权重的和。



一颗树定义为
$$
f_k = w_{q(x_i)} \tag{8}
$$
$q(x_i)$ 表示，将一个样本 $x_i$ 代入这棵树，被分到的叶子结点的编号



### 5. 树的复杂度

树的复杂度由以下两方面组成：

- 叶子结点的数量 T
- 叶子结点的权重向量的正则项



此处为 $L_2$ 正则项
$$
Ω(f_t) = \gamma T + \frac{1}{2}\lambda \sum^T_{j=1}w^2_j \tag{9}
$$


### 6. 对样本落入的叶子结点进行分组

 将属于第 j 个叶子结点的所有样本 xi , 划入到一个叶子结点样本集中：


$$
I_j = \{i|q(x_i) = j\}
$$
将 (8) (9) 代入 (7) 
$$
\begin{align}
Obj^{(t)} &= \sum^n_{i=1}\Big{[}g_if_t(x_i) + \frac{1}{2}h_if_t(x_i)^2) \Big{]} + Ω(f_t)\\

&= \sum^{n}_{i=1} \Big{[} g_iw_{q(x_i)} + \frac{1}{2}h_iw_{q(x_i)}^2 \Big{]} + \gamma T + \frac{1}{2}\lambda \sum^T_{j=1}w^2_j\\

&= \sum^{n}_{i=1}  g_iw_{q(x_i)} + \sum^{n}_{i=1}\frac{1}{2} h_iw_{q(x_i)}^2  + \gamma T + \frac{1}{2}\lambda \sum^T_{j=1}w^2_j\\

&= \sum^T_{j=1}\big( w_j*\sum_{i\in I_j}g_i \big) + \frac{1}{2}\sum^T_{j=1}\big( w_j^2*\sum_{i\in I_j}h_i \big) + \gamma T + \frac{1}{2}\lambda \sum^T_{j=1}w^2_j\\

&= \sum^T_{j=1} \Big[  w_j*\sum_{i\in I_j}g_i + \frac{1}{2}w_j^2(\sum_{i\in I_j}h_i + \lambda)   \Big] + \gamma T \tag{10}

\end{align}
$$
被分到同一个叶子结点里面的样本，对应的叶子结点的权重是相同的，这个样本集里面的所有样本的权重之和 = 所有被分到同一组的样本*该组的权重之和



令 $G_i= \sum_{i\in I_j}g_i$ ，$H_i= \sum_{i\in I_j}h_i$

- $G_j$ ：**叶子结点 j 所包含样本**的`一阶偏导数`累加之和，是一个常量；
- $H_j$ ：**叶子结点 j 所包含样本**的`二阶偏导数`累加之和，是一个常量；



原式：
$$
Obj^{(t)} = \sum^T_{j=1} \Big[  G_jw_j + \frac{1}{2}(H_j + \lambda)w_j^2   \Big] + \gamma T \tag{11}
$$


此时 $F(w_j) = G_jw_j + \frac{1}{2}(H_j + \lambda)w_j^2$ 就是一个自变量为 $w_j$ 的二次函数

令 $F'(w_j) = 0$ 得：
$$
w_j^* = -\frac{G_j}{H_j + \lambda}\\
F(w_j^*) = -\frac{1}{2}\frac{G_j^2}{H_j + \lambda} \tag{12}
$$
此时损失函数达到最小值，所以最优树结构对应的目标函数是：
$$
Obj^{(t)} = -\frac{1}{2}\sum^T_{j=1}\frac{G_j^2}{H_j + \lambda} + \gamma T \tag{13}
$$
由于 $\gamma$ 是认为给定的，所以此时树的目标值，就只和叶子结点的数量由关系了



### 7. 树结点的分裂

一个树结点可以看成是只有一颗只有根结点的树，所以一个树结点的分数为
$$
Score = -\frac{1}{2} \frac{G_j^2}{H_j + \lambda} + \gamma
$$
xgboost规定，作为基模型的树是二叉树，所以一个结点，只能分为两个子结点，与决策树一样，我们要定义一个指标来衡量，按照某一个特征进行分支的前后的增益量

增益量定义：
$$
\begin{aligned}
Gain &= Obj_{L+R} - (Obj_L + Obj_R)\\
&= \frac{1}{2} \Big[ \frac{G^2_L}{H_L+\lambda} + \frac{G^2_LR}{H_R+\lambda} - \frac{(G_L + G_R)^2}{H_L+H_R\lambda}  \Big] - \gamma
\end{aligned}
$$
当 Gain > 0 的时候，即分支后的损失相比于分支前下降了，才会考虑进行分支

### 8. 寻找最优分裂结点

寻找最佳分割点的大致步骤如下：

- 遍历每个结点的每个特征；
- 对每个特征，按特征值大小将特征值排序；
- 线性扫描，找出每个特征的最佳分裂特征值；
- 在所有特征中找出最好的分裂点（分裂后增益最大的特征及特征值）
- 

为了更快的寻找最佳的分裂结点,xgboost采用了一些更好的方法:

- 特征预排序 + 缓存: xgboost在训练之前,预先对每一个特征的特征值进行预排序,保存为block结构,后面迭代的时候重复使用这个结构,减小计算量
- 分位点近似法：对每个特征按照特征值排序后，采用类似分位点选取的方式，仅仅选出常数个特征值作为该特征的候选分割点，在寻找该特征的最佳分割点时，从候选分割点中选出最优的一个。
- 并行查找：由于各个特性已预先存储为block结构，XGBoost支持利用多个线程并行地计算每个特征的最佳分割点，这不仅大大提升了结点的分裂速度，也极利于大规模训练集的适应性扩展。



### 9. 树的停止生长

1. Gain < 0
2. 树达到max_depth
3. 如果一个结点的样本权重低于阈值, 此节点停止分裂



参考

https://mp.weixin.qq.com/s?__biz=MzI1MzY0MzE4Mg==&mid=2247485277&idx=2&sn=a47e5e94a86fb215fe11b0ea867b2732&chksm=e9d0179cdea79e8a41f52f3626a43f3ebc08b717290b9d27b3860586af47393229c27cda2134&scene=0&xtrack=1&key=48335815060ff6d3f8e1dd5859e30e67c99d9af888f5516083ddd6967fcf937f53b56cbd19362438b8d5fb24b209dc9d7d876db7fefd088a6ecad35534f1c6ee2321363a265d4072bb733c4163e961db&ascene=14&uin=MTE4NzIwNTU3Nw%3D%3D&devicetype=Windows+10&version=62060834&lang=zh_CN&pass_ticket=Hkdhk5JarBYQf2g0n8fM3spT0%2BcVSzwfyFA6Z8NR%2BUKou2fl0U5PD1WkIjiameeJ



https://www.lizenghai.com/archives/3192.html

![1571918824138](C:\Users\Liang\AppData\Roaming\Typora\typora-user-images\1571918824138.png)

