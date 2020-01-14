# Unsupervised Learning for URL Classification

## 已经尝试的Methods
* 对已经labeled的data进行监督学习，但是这样的prediction几乎全是others，
只有artical和encyclopedia准确率比较高，原因还是再有数据太少了。
* 对所有的data进行聚类，但是如何把聚类出来的10个类和原本data中的类对应
上也是个问题。

## 接下来的方向
主要的问题还是数据太少，感觉只能做半监督学习

* 先用标注好的数据来进行训练，然后利用训练好的学习器找出未标注数据中能
对性能改善最大的数据来询问“专家”。这样只需要专家标注比较少的数据就能得
到较强的学习器了。
* 半监督聚类：在非监督学习的基础上引入监督信息来优化性能,约束 k 均值（Constrained k-means）算法
* 自编码聚类算法--DEC (Deep Embedded Clustering)
* 其他半监督学习方法

# Using Image and HTML for classification
还是老问题，label太少，目前还是先做好feature engineering丢进网络
里试试吧

# How to Achieve Innovations?
感觉只做分类真的8太行，想想还有啥其他比较fancy的功能可以做吧

mark两个去年的链接
* [Best Project](https://github.com/forwchen/celeste)
* [Most Innovative](https://github.com/Google-winter-camp/ZJU)