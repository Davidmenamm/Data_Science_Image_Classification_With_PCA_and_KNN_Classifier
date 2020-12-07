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
import pandas as pd

# constants
imgHeight = 256
imgWidth = 256

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

# RUN PCA
nComp = 80
pca = PCA(n_components=nComp, whiten=True).fit(np.array(imgVectorsTrain))

# literal 2: Scree Plot to see ideal number of components
# corresponding eigen values
plt.figure('Scree Plot')
eVals = np.square(pca.singular_values_)
plt.plot(range(1, len(eVals)+1), eVals, marker='.',
         label='Scree Plot for PCA')
plt.xlabel('Component Number')
plt.ylabel('Eigen Values')
# show the legend
plt.legend()
# show the plot
plt.show()

# literal 6: Classification
# transform train data to PCA, principle axis
pca_imgVectorsTrain = pca.transform(np.array(imgVectorsTrain))
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


# find best k for knn classification
typeDist = 2  # euclidean
knnTestAcc = []
for kValue in range(2, 20):
    # build and train model that uses euclidean distance
    clfTest = KNeighborsClassifier(n_neighbors=kValue, p=typeDist)
    clfTest.fit(pca_imgVectorsTrain, labelsTrain)
    # predicted labels
    pred_labelsKnn = clfTest.predict(pca_imgVectorsTest)
    knnTestAcc.append(metrics.accuracy_score(labelsTest, pred_labelsKnn))
# plot best k
plt.figure('K vs Accuracy')
plt.subplot(111)
plt.plot(range(2, 20), knnTestAcc, marker='.')
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.show()


# literal 5: Print eigen vector matrix!
with open(r'data\results\eigenVectMatrix.txt', 'w', encoding="utf-8") as f:
    eigenVectMatrix = pca.components_
    pdEigenVectMatrix = pd.DataFrame(eigenVectMatrix.transpose())
    f.write('Eigen Vector Matrix\n\n')
    f.write(str(pdEigenVectMatrix) + '\n')
    nRows, nCols = pca.components_.transpose().shape
    pcColNames = []
    for colNum in range(len(eigenVectMatrix)):
        pcColNames.append(f'PC{colNum}')

# literal 3: accumulated variance for chosen eigen vectors in 2) literal
dictAccVariance = {}
pcaVariance = pca.explained_variance_ratio_
countVar = 0
accVariance = 0
accVarianceArr = []
for var in pcaVariance:
    countVar += 1
    accVariance += var
    accVarianceArr.append(accVariance)
    dictAccVariance[f'PC-{countVar}'] = accVariance
# print results
strAccVariance = 'Accumulated variance for selected Principle Components\n\n'
for k, v in dictAccVariance.items():
    strAccVariance += f'{k} -> {v}\n'
with open(r'data\results\AccVariance.txt', 'w', encoding="utf-8") as f:
    f.write(strAccVariance)
# plot accumulated variance
plt.figure('Accumulated variance vs components')
plt.subplot(111)
plt.plot(range(1, nComp+1), accVarianceArr, marker='.')
plt.hlines(0.95, 0, nComp, colors='r')
plt.xlabel('Acc. Variance')
plt.ylabel('Number of Components')
plt.show()

# literal 1: Graph the eigen face of the mean of the initial matrix, before applyin PCA
meanVect = pca.mean_
plt.figure('Mean Vector Face')
plt.subplot(111)
plt.imshow(meanVect.reshape((imgHeight, imgWidth)), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()

# literal 3: Graph all eigen faces for selected principle components
plt.figure('Eigen Faces')
graphRows = 8
graphCols = 10
for i in range(graphRows * graphCols):
    plt.subplot(graphRows, graphCols, i + 1)
    plt.imshow(eigenVectMatrix[i].reshape(
        (imgHeight, imgWidth)), cmap=plt.cm.gray)
    # plt.title(f'PC-{i}')
    plt.xticks(())
    plt.yticks(())
plt.show()
