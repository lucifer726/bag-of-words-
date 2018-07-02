# bag-of-words-
基于opencv-python的sift、kmeans、bow图像检索
需要配置opencv、sklearn、scipy、numpy

创建两个文件夹就行

默认图像训练文件名为dataset
我用的是101_ObjectCategories图片集
所以在读入图片时做了更改

默认查找图像文件名为query

用命令行执行python findFeatures.py -t dataset/  开始生成模型

用命令行执行python search.py -i query/target.jpg  查找目标图片
