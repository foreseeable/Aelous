# Unsupervised Learning for Webpage Classification

## 已经尝试的Methods
* 对已经labeled的data进行监督学习，但是这样的prediction几乎全是others，
只有artical和encyclopedia准确率比较高，原因还是再有数据太少了。
* 对所有的data进行聚类，但是如何把聚类出来的10个类和原本data中的类对应
上也是个问题
* 输入image，使用encoder-decoder模型现找出data representation，然后用
监督学习找到hidden-representation到label的关系，前一步已经完成。

## 接下来的方向
主要的问题还是数据太少，感觉只能做半监督学习

* 先用标注好的数据来进行训练，然后利用训练好的学习器找出未标注数据中能
对性能改善最大的数据来询问“专家”。这样只需要专家标注比较少的数据就能得
到较强的学习器了。
* 半监督聚类：在非监督学习的基础上引入监督信息来优化性能,约束 k 均值（Constrained k-means）算法
* 输入html进行分类。

# Using Image and HTML for classification
到时候做成应用的话肯定是优先使用html的，但是html的输入和feature extraction又是
一个问题。

# How to Achieve Innovations?
感觉只做分类真的8太行，想想还有啥其他比较fancy的功能可以做吧？
目前想到了几个可能的应用：
* 做成浏览器插件，然后浏览器检测到不同类型的网页时会做不同的优化，就
像switch Omega 一样。并且这个是已经有类似的应用的：wikiwand
* 浏览器收集用户浏览的网页类型，精准推送相关的广告。比如如果artical和
wiki比较多的话，说明用户比较喜欢学习...something like that
* 判断是否是entity，从而进行一个搜索的排序

mark两个去年的链接
* [Best Project](https://github.com/forwchen/celeste)
* [Most Innovative](https://github.com/Google-winter-camp/ZJU)