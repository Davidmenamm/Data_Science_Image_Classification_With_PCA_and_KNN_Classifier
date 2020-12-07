# Test program

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


# Load data
people = fetch_lfw_people(min_faces_per_person=100)

_, h, w = people.images.shape
X = people.data
y = people.target
target_names = people.target_names

print('data is ', X[0])
print('data size is ', X.shape)
print('size , ', h, w)
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print('xtrain is ', X_train, type(X_train), type(X_train[0]))


# Compute a PCA
n_components = 100
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# apply PCA transformation
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# train a neural network
print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256,
                    verbose=True, early_stopping=True).fit(X_train_pca, y_train)


# classification report
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))


# Visualization
def plot_gallery(images, titles, h, w, rows=5, cols=5):
    plt.figure()
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
    plt.show()


def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = target_names[y_pred[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)


print('target names ', target_names, y_pred)
prediction_titles = list(titles(y_pred, y_test, target_names))
print(prediction_titles)

print('images, ', X_test, X_test.shape)
plot_gallery(X_test, prediction_titles, h, w)
