# Aelous

原始总共10833个html文件

有4个解析器无法正常解析的文件恰好也是不在我们数据范围内的：5938；8957；4786；4787

处理后得到10829个txt文件（提取的内容包括title和p<paragraph>）
  
  其中有102个文件（见``problem_files.txt``）因遇到html中不合法的符号出现报错，目前的处理方式是直接停止解析，保留报错前已提取的内容

``processed_html``中保存的是从html中提取的内容

``tokenized_html``中保存的是分词之后的内容

``vectors``中保存的是已转换好的vectors，load方式可参照``load_vectors.ipynb``

``doc2vec.ipynb``是doc2vec的全部代码

``html_parser_trimmed``是最初parse html的代码

训出来的models似乎因为太大push不了，不过训起来很快（如果需要的话）
