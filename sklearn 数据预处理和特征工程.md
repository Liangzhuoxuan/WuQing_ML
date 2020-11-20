*这个笔记就是根据数据预处理和特征工程的流程进行设置的*

[TOC]

# sklearn 数据预处理和特征工程

## 1 数据预处理 Preprocessing & Impute

### 1.1 **数据无量纲化**

将不同规格的数据转换到同一规格，或不同分布的数据转换到某个特定分布，譬如梯度和矩阵为核心的算法中，譬如逻辑回归，支持向量机，神经网络，无量纲化可以加快求解速度；而在距离类模型，譬如K近邻，K-Means聚类中，无量纲化可以帮我们提升模型精度，避免某一个取值范围特别大的特征对距离计算造成影响。（一个特例是决策树和树的集成算法们，对决策树我们不需要无量纲化，决策树可以把任意数据都处理得很好。）



无量纲化包括：**中心化**和**缩放处理**，中心化的本质是让数据进行平移，缩放的本质是将数据固定到某一个范围，对指化也是一种缩放处理。

- preprocessing.MinMaxScaler

**数据的归一化**：将数据收敛到 [0, 1] 之间
$$
x^* = \frac{x - min(x)}{max(x) - min(x)}
$$

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
result_ = scaler.fit_transform(data)
 
scaler.inverse_transform(result)
# 将归一化后的数据进行反转，有的时候不小心将原数据传入了，没传temp，就可以用反转获得原来的数据
```

- preprocessing.StandardScaler

**数据的标准化**：将数据变为服从均值为 0，方差为 1的正态分布的数据。
$$
x^* = \frac{x - μ}{σ}
$$

```python
from sklearn.preprocessing import StandardScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
 
scaler = StandardScaler()                           #实例化
scaler.fit(data)                                    #fit，本质是生成均值和方差
 
scaler.mean_                                        #查看均值的属性mean_
scaler.var_                                         #查看方差的属性var_
 
x_std = scaler.transform(data)                      #通过接口导出结果
 
x_std.mean()                                        #导出的结果是一个数组，用mean()查看均值
x_std.std()                                         #用std()查看方差
 
scaler.fit_transform(data)                          #使用fit_transform(data)一步达成结果
 
scaler.inverse_transform(x_std)                     #使用inverse_transform逆转标准化
```



注意，MinMaxScaler 和 StandardScaler 在 fit 的时候，会忽略里面的 NaN 空值，sklearn 里面的 fit 接口几乎都要传入二维数组，如果是一维要 reshape(-1, 1) 一下。

- StandardScaler 和 MinMaxScaler 选哪个？

看情况。大多数机器学习算法中，会选择StandardScaler来进行特征缩放，因为MinMaxScaler对异常值非常敏感。在PCA，聚类，逻辑回归，支持向量机，神经网络这些算法中，StandardScaler往往是最好的选择。

MinMaxScaler在不涉及距离度量、梯度、协方差计算以及数据需要被压缩到特定区间时使用广泛，比如数字图像处理中量化像素强度时，都会使用MinMaxScaler将数据压缩于[0,1]区间之中。

建议先试试看StandardScaler，效果不好换MinMaxScaler。

其他的缩放处理：在希望压缩数据，却不影响数据的稀疏性时（不影响矩阵中取值为0的个数时），我们会使用MaxAbsScaler；在异常值多，噪声非常大时，我们可能会选用分位数来无量纲化，此时使用RobustScaler。

### 1.2 缺失值处理



直接用 pandas 填就完事了。

[缺失值处理博客](https://mp.weixin.qq.com/s?__biz=MzI2NjkyNDQ3Mw==&mid=2247490808&idx=2&sn=8fa81bad29f4b702be11b6af6e564f1a&chksm=ea87e42eddf06d387ecc65128c21ad31b40700d8f8a27b160c2a93a6d0b099fbb483deb47b48&scene=0&xtrack=1&key=f9b43649f6789a0c0efbba3335c8290b66cf3d82ad0a897c3c9b70fb2a5203e504f5a5e3064bb17b315b641d0f335e8390b0f31876cfd32de2d0905425e650c329a172d7f1f5a80a41c5e549bd84c4fc&ascene=14&uin=MTE4NzIwNTU3Nw%3D%3D&devicetype=Windows+10&version=62060834&lang=zh_CN&pass_ticket=heGZi%2BLUkRnGo8xAktF4SuoQcrZWQ8LlvydduFW7gSI7tGXH%2B1n6UpIlWwooTdHw)。



### 1.3  处理离散型特征，编码 与 哑变量

机器学习中的很多算法只能处理数值型特征，不能处理字符串型的，此时我们要对这些数据进行编码

- preprocessing.LabelEncoder

```python
from sklearn.preprocessing import LabelEncoder
data.iloc[:,-1] = LabelEncoder().fit_transform(data.iloc[:,-1])
```



- preprocessing.OrdinalEncoder：特征专用

  ```python
  from sklearn.preprocessing import OrdinalEncoder
  
  OrdinalEncoder().fit(data_.iloc[:,1:-1]).categories_
   
  data_.iloc[:,1:-1] = OrdinalEncoder().fit_transform(data_.iloc[:,1:-1])
  ```

  

- preprocessing.OneHotEncoder 独热编码，创建哑变量

先看三种离散型变量，如 Titanic 里面的 Embarked（'S'，'C'，'Q'），学历（小学，中学，大学），体重（>45kg，>90kg，>135kg），对于体重这个离散型特征来说，各个取值之间是可以相互计算的，如：120kg - 45kg = 90kg，分类之间可以通过数学计算互相转换。这是**有距变量**。

但是在对这三种离散型变量进行编码的时候，都会转化为[0, 1, 2]，这三个数字在算法看来是连续且可以计算的，是有大小之分的，有着可以进行加减乘除的联系，所以 Embarked，学历 这两个特征都会被误认为是体重这样的特征，会给算法传达了一些不准确的信息，而这会影响我们的建模。

**对于名义变量，我们要用哑变量的方式来处理**

![1566126888710](C:\Users\Liang\AppData\Roaming\Typora\typora-user-images\1566126888710.png)

用了独热编码之后，每一个离散值都有其对应的唯一编码，这样，他们之间就呈现不可计算的性质。

```python
from sklearn.preprocessing import OneHotEncoder
X = data.iloc[:,1:-1]
 
enc = OneHotEncoder(categories='auto').fit(X)
result = enc.transform(X).toarray()
result

enc.get_feature_names()# 返回每一个经过哑变量后生成稀疏矩阵列的名字

# 将编码后的特征合并到表中
# axis=1,表示跨行进行合并，也就是将两表左右相连，如果是axis=0，就是将量表上下相连
newdata = pd.concat([data,pd.DataFrame(result)],axis=1)

# 删掉原来的特征
newdata.drop(["Sex","Embarked"],axis=1,inplace=True)
```

### 1.4 处理连续型特征：二值化 与 分段

- sklearn.preprocessing.Binarizer

大于阈值的值映射为1，而小于或等于阈值的值映射为0。

```python
from sklearn.preprocessing import Binarizer

X = data_2.iloc[:,0].values.reshape(-1,1)
```



- preprocessing.KBinsDiscretize

这是将连续型变量划分为分类变量的类，能够将连续型变量排序后按顺序分箱后编码。总共包含三个重要参数：

| 参数     | 含义&输入                                                    |
| -------- | ------------------------------------------------------------ |
| n_bins   | 每个特征中分箱的个数，默认5，一次会被运用到所有导入的特征    |
| encode   | 编码的方式，默认“onehot” "onehot"：做哑变量，之后返回一个稀疏矩阵，每一列是一个特征中的一个类别，含有该类别的样本表示为1，不含的表示为0 “ordinal”：每个特征的每个箱都被编码为一个整数，返回每一列是一个特征，每个特征下含有不同整数编码的箱的矩阵"onehot-dense"：做哑变量，之后返回一个密集数组。 |
| strategy | 用来定义箱宽的方式，默认"quantile" "uniform"：表示等宽分箱，即每个特征中的每个箱的最大值之间的差为(特征.max() - 特征.min())/(n_bins) "quantile"：表示等位分箱，即每个特征中的每个箱内的样本数量都相同"kmeans"：表示按聚类分箱，每个箱中的值到最近的一维k均值聚类的簇心得距离都相同 |

```python
from sklearn.preprocessing import KBinsDiscretizer
 
X = data.iloc[:,0].values.reshape(-1,1) 
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
est.fit_transform(X)
 
#查看转换后分的箱：变成了一列中的三箱
set(est.fit_transform(X).ravel())
 
est = KBinsDiscretizer(n_bins=3, encode='onehot', strategy='uniform')
#查看转换后分的箱：变成了哑变量
est.fit_transform(X).toarray()
```



## 2 特征选择 feature_selection

| 特征提取                                                     | 特征创建                                       | 特征选择                                                     |
| ------------------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------ |
| 从当前各个特征中提取出一些特征，如一个淘宝商品的名称，从中提取产品类型，产品颜色等 | 把现有特征进行组合，或互相计算，得到新的特征。 | 从现有的特征中选择有意义，对模型有帮助的特征，避免将所有特征都拿去训练模型 |

**特征工程的第一步是理解业务**

对于一些无法理解业务的数据，如泰坦尼克号这种，我们有四种方法可以用来选择特征：过滤法，嵌入法，包装法，和降维算法。

### 2.1 Filter过滤法

![1566128723443](C:\Users\Liang\AppData\Roaming\Typora\typora-user-images\1566128723443.png)

#### 2.1.1 方差过滤

通过特征本身的方差来筛选特征，如果一个特征本身的方差很小，就表示样本在这个特征上基本没有差异，这样的特征对样本的区分没啥意义，所以无论特征工程接下来要干哈，都要先消除方差很小的特征

- VarianceThreshold

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold()                      #实例化，不填参数默认方差为0
X_var0 = selector.fit_transform(X)                  #获取删除不合格特征之后的新特征矩阵

# 我们希望留下一半的特征，那可以设定一个让特征总数减半的方差阈值，只要找到特征方差的中位数，再将这个中位数作为参数threshold的值输入就好了

import numpy as np
# X.var()#每一列的方差
X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
 
X.var().values
 
np.median(X.var().values)
 
X_fsvar.shape

#若特征是伯努利随机变量，假设p=0.8，即二分类特征中某种分类占到80%以上的时候删除特征
X_bvar = VarianceThreshold(.8 * (1 - .8)).fit_transform(X)
X_bvar.shape
```



最近邻算法KNN，单棵决策树，支持向量机SVM，神经网络，回归算法，都需要遍历特征或升维来进行运算，所以他们本身的运算量就很大，需要的时间就很长，因此**方差过滤这样的特征选择对他们来说就尤为重要**。但对于不需要遍历特征的算法，比如随机森林，它随机选取特征进行分枝，本身运算就非常快速，因此特征选择对它来说效果平平。

因此，过滤法的主要对象是：**需要遍历特征或升维的算法们**，而过滤法的**主要目的**是：**在维持算法表现的前提下，帮助算法们降低计算成本**



超参数 threshold 的选择：

**现实中，我们只会使用阈值为0或者阈值很小的方差过滤，来为我们优先消除一些明显用不到的特征，然后我们会选择更优的特征选择方法继续削减特征数量。**

#### 2.1.2 相关性过滤

我们希望选出与标签相关且有意义的特征，因为这样的特征能够为我们提供大量信息。如果特征与标签无关，那只会白白浪费我们的计算内存，可能还会给模型带来噪音。在sklearn当中，我们有三种常用的方法来评判特征与标签之间的相关性：卡方，F检验，互信息。

##### 2.1.2.1 卡方过滤

方过滤是专门针对**离散型标签（即分类问题）**的相关性过滤。卡方检验的本质是推测两组数据之间的差异，其检验的原假设是”两组数据是相互独立的”。卡方检验返回卡方值和P值两个统计量，其中卡方值很难界定有效的范围，而p值，我们一般使用0.01或0.05作为显著性水平，即p值判断的边界，p值小于显著性水平，即特征和标签是相关联的。

方法：卡方检验类feature_selection.chi2 计算每个非负特征与标签之间的卡方统计量，并按照卡方统计量对特征进行排名，所以**如果特征有负值，要对特征先进行归一化**，再结合feature_selection.SelectKBest 来选出前K个分数最高的特征的类，我们可以借此除去最可能独立于标签，与我们分类目的无关的特征。



```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
 
#假设在这里我一直我需要300个特征
X_fschi = SelectKBest(chi2, k=300).fit_transform(X_fsvar, y)
X_fschi.shape

# 卡方值，p值
chivalue, pvalues_chi = chi2(X_fsvar,y) # 输入 特征矩阵 和 标签
 
chivalue
 
pvalues_chi
 
#k取多少？我们想要消除所有p值大于设定值，比如0.05或0.01的特征：
k = chivalue.shape[0] - (pvalues_chi > 0.05).sum()
```



##### 2.1.2.2 F检验

F检验，方差齐次性检验，是用来捕捉每个特征与标签之间的线性关系的过滤方法。它即可以做回归也可以做分类，因此包含feature_selection.f_classif（F检验分类）和feature_selection.f_regression（F检验回归）两个类。其中F检验分类用于标签是离散型变量的数据，而F检验回归用于标签是连续型变量的数据。

和卡方检验一样，这两个类需要和类SelectKBest连用，F检验在数据服从正态分布时效果会非常稳定，因此**如果使用F检验过滤，我们会先将数据转换成服从正态分布的方式**。

```python
from sklearn.feature_selection import f_classif
 
F, pvalues_f = f_classif(X_fsvar,y)
 
pvalues_f
 
k = F.shape[0] - (pvalues_f > 0.05).sum()
```



##### 2.1.2.3 互信息法

3.1.2.4 互信息法互信息法是用来捕捉每个特征与标签之间的任意关系（包括线性和非线性关系）的过滤方法。和F检验相似，它既可以做回归也可以做分类，并且包含两个类feature_selection.mutual_info_classif（互信息分类）和feature_selection.mutual_info_regression（互信息回归）。这两个类的用法和参数都和F检验一模一样，不过互信息法比F检验更加强大，F检验只能够找出线性关系，而互信息法可以找出任意关系。

互信息法不返回p值或F值类似的统计量，它返回“每个特征与目标之间的互信息量的估计”，这个估计量在[0,1]之间取值，为0则表示两个变量独立，为1则表示两个变量完全相关。

```python
from sklearn.feature_selection import mutual_info_classif as MIC
 
result = MIC(X_fsvar,y)
 
k = result.shape[0] - sum(result <= 0)
```



**最好先用方差过滤，然后就用 互信息法**



## 2.2 Embedded嵌入法

## 2.3 包装法



