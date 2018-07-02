#python findFeatures.py -t dataset/

import argparse as ap
import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing


# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = "dataset\\"   #训练样本文件夹路径
training_names = os.listdir(train_path)
numWords = 64  # 聚类中心数

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []  # 所有图片路径
ImageSet = {}
for name in training_names:
    ls = os.listdir(train_path + "/" + name)
    ImageSet[name] = len(ls)
    for training_name in ls[:int(len(ls) / 3)]:
        image_path = os.path.join(train_path + name, training_name)
        image_paths += [image_path]

# Create feature extraction and keypoint detector objects
sift_det=cv2.xfeatures2d.SIFT_create()
# List where all the descriptors are stored
des_list=[]  # 特征描述


for name, count in ImageSet.items():
    dir = train_path + name
    print("从 " + name + " 中提取特征")
    trainNum = int(count / 3)
    for i in range(1, trainNum + 1):
        filename = dir + "\image_" + str(i).rjust(4, '0') + ".jpg"
        img=cv2.imread(filename)
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        kp,des=sift_det.detectAndCompute(gray,None)
        des_list.append((image_path, des))



# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
print('生成向量数组')
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

# Perform k-means clustering
print ("开始 k-means 聚类: %d words, %d key points" %(numWords, descriptors.shape[0]))
voc, variance = kmeans(descriptors, numWords, 1) 

# Calculate the histogram of features
im_features = np.zeros((len(image_paths), numWords), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Perform L2 normalization
im_features = im_features*idf
im_features = preprocessing.normalize(im_features, norm='l2')

print('保存词袋模型文件')
joblib.dump((im_features, image_paths, idf, numWords, voc), "bow.pkl", compress=3)
