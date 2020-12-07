# Client running the program

# read files images iteratively
import os
import glob
from PIL import Image
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict

# Build training set and training labels
# Labels are based on the persons name, first letter of the picture file
labelsTrain = []
labelCode = {
    'KA': 0,
    'KL': 1,
    'KM': 2,
    'KR': 3,
    'MK': 4,
    'NA': 5,
    'NM': 6,
    'TM': 7,
    'UY': 8,
    'YM': 9,
}
imgVectorsTrain = []
# go through training images
for filepath in glob.glob(os.path.join(r'data\train_dataset', '*.jpg')):
    # Assign labels
    labelsTrain.append(labelCode[filepath[19:21]])
    # Create vector grayscale images
    PIL_img = Image.open(filepath).convert('L')
    # width, height = PIL_img.size
    imgVectorsTrain.append(np.array(PIL_img).flatten().tolist())

# run PCA
nComp = 78
pca = PCA(n_components=nComp, whiten=True).fit(np.array(imgVectorsTrain))

# transform train data to PCA, principle axis
pca_imgVectorsTrain = pca.transform(np.array(imgVectorsTrain))

# literal 6: Classification
typeDist = 2  # euclidean
k = 8  # number of neighbours
# build and train model that uses euclidean distance
clf = KNeighborsClassifier(n_neighbors=k, p=typeDist)
clf.fit(pca_imgVectorsTrain, labelsTrain)
# go through test data set images
imgVectorsTest = []
testImgNames = []
labelsTest = []
for filepath in glob.glob(os.path.join(r'data\test_dataset', '*.jpg')):
    # test correct labels
    labelsTest.append(labelCode[filepath[18:20]])
    # Create vector grayscale images
    PIL_img = Image.open(filepath).convert('L')
    # file shorter name
    fileName = filepath.split('\\')[2]
    testImgNames.append(fileName)
    # width, height = PIL_img.size
    imgVectorsTest.append(np.array(PIL_img).flatten().tolist())
# transform test data with PCA
pca_imgVectorsTest = pca.transform(np.array(imgVectorsTest))
# predicted labels
pred_labels = clf.predict(pca_imgVectorsTest)
# results
resultDict = defaultdict(list)
invLabCode = dict((y, x) for x, y in labelCode.items())
for label, fileName in zip(pred_labels, testImgNames):
    resultDict[invLabCode[label]].append(fileName)
# result dictionary to string
strResult = 'PCA, to aid the process of image classification\n'
strResult += f'Accuracy: {metrics.accuracy_score(labelsTest, pred_labels)}\n\n'
for key, val in resultDict.items():
    strResult += str(key) + '\n'
    for v in val:
        strResult += '\t\t' + str(v) + '\n'
# print to file categories
with open(r'data\results\categories.txt', 'w', encoding="utf-8") as f:
    f.write(strResult)

# literal 5: Print eigen vector matrix!
with open(r'data\results\eigenVectMatrix.txt', 'w', encoding="utf-8") as f:
    f.write('Eigen Vector Matrix\n\n')
    f.write(str(pca.components_.transpose()))
    f.write(str(pca.components_.transpose().shape()))

# literal 3: accumulated variance for chosen eigen vectors in 2) literal
dictAccVariance = {}
pcaVariance = pca.explained_variance_ratio_
countVar = 0
accVariance = 0
for var in pcaVariance:
    countVar += 1
    accVariance += var
    dictAccVariance[f'PC{countVar}'+str(countVar)] = accVariance
# print results
strAccVariance = 'Accumulated variance for selected Principle Components\n\n'
for k, v in dictAccVariance.items():
    strAccVariance += f'{k} -> {v}\n'
with open(r'data\results\AccVariance.txt', 'w', encoding="utf-8") as f:
    f.write(strAccVariance)

# test printings
print('0\t', pca.components_)
print('1\t', pca.components_.shape)
print('2\t', pca.explained_variance_)
print('3\t', pca.explained_variance_ratio_,
      np.sum(pca.explained_variance_ratio_))
print('4\t', pca.singular_values_)
print('5\t', pca.mean_)
print('6\t', pca.n_components_)
print('7\t', pca.n_features_)
print('8\t', pca.noise_variance_)
