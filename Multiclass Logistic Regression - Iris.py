import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Loading the dataset
iris = datasets.load_iris()

# take first two columns of iris data (sepal length, sepal width)
X = iris.data[:,:2]

# take label for each iris (there are 3, our classes)
y = iris.target

print("num samples, num features =", X.shape)

plt.figure(figsize=(8,6))
plt.rcParams['figure.figsize'] = [6, 4]

# seperate X by class and get both columns where X matches the set y class
plt.plot(X[y==0,0], X[y==0,1], 'o', markerfacecolor=(1,0,0,1), markeredgecolor='black', label="class 0")
plt.plot(X[y==1,0], X[y==1,1], 'o', markerfacecolor=(0,1,0,1), markeredgecolor='black', label="class 1")
plt.plot(X[y==2,0], X[y==2,1], 'o', markerfacecolor=(0,0,1,1), markeredgecolor='black', label="class 2")
# remember this a 3-class classification problem!

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.grid()
plt.legend()

# split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.1, shuffle=True)

# create logistic regression object
clf = LogisticRegression(random_state=0)

# fit the model on the training data ONLY
clf.fit(X_train, y_train)


# predict data for training and testing data and compare accuracy
y_pTr = clf.predict(X_train)
y_pTe = clf.predict(X_test)
train_acc = np.mean(y_train == y_pTr)
test_acc = np.mean(y_test == y_pTe)

print("Training Accuracy: "+ str(train_acc), "Test Accuracy: "+ str(test_acc))


### CODE TAKEN FROM SKLEARN IRIS DEMO ###

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
plt.figure(figsize=(8,6))
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, shading='auto')

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())


# get probablity for each class in every row
yhat_probs_test = reg.predict_proba(X_test)
np.set_printoptions(precision=2, suppress=True)
print(y_pTe.reshape(-1, 1).shape, yhat_probs_test.shape)
# [class, prob_of_class_1, prob_of_class_2, prob_of_class_3]
print(np.hstack([y_pTe.reshape(-1,1), yhat_probs_test]))

plt.show()
