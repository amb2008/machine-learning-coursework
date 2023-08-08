import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model, preprocessing
from sklearn.linear_model import LogisticRegression
from matplotlib.textpath import FontProperties

#call desired function at end

# Set list containing header names of CSV
names = ['id','thick','size','shape','marg','cell_size','bare',
         'chrom','normal','mit','class']

# Read CSV into pandas dataframe
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/' +'breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=names,na_values='?',header=None)

# Drop NA values will nul
df = df.dropna()

# Get column from DF containing the class (label) of the cell
yraw = np.array(df['class'])
BEN_VAL = 2   # value in the 'class' label for benign samples
MAL_VAL = 4   # value in the 'class' label for malignant samples

# set y to list where maling is 1 and baning is 0
y = (yraw == MAL_VAL).astype(int)

# get vector with true where cells are baning and false for maning
Iben = (y==0)

# get vector with true where cells are maning and false for baning
Imal = (y==1)

# Create matrix of just size and marg
xnames =['size','marg']
X = np.array(df[xnames])

def plotScatter():
  plt.figure(figsize=(8,6))
  
  # get index of all malign cells and plot their size and marg from X matrix
  plt.plot(X[Imal,0],X[Imal,1],'r.')
  
  # get index of all banign cells and plot their size and marg from X matrix
  plt.plot(X[Iben,0],X[Iben,1],'g.')
  plt.xlabel(xnames[0], fontsize=16)
  plt.ylabel(xnames[1], fontsize=16)
  plt.ylim(0,14)
  plt.legend(['malign','benign'],loc='upper right')



def plot_weighted_scatter():
  # Compute the bin edges for the 2d histogram
  # pull out all elements of size data and make them float
  x0val = np.array(list(set(X[:,0]))).astype(float)

  # pull out all elements of marg data and make them float
  x1val = np.array(list(set(X[:,1]))).astype(float)

  #compare amounts of each class
  x0, x1 = np.meshgrid(x0val,x1val)
  x0e= np.hstack((x0val,np.max(x0val)+1))
  x1e= np.hstack((x1val,np.max(x1val)+1))

  # Make a plot for each class
  yval = list(set(y))
  color = ['g','r']
  plt.figure(figsize=(8,6))
  for i in range(len(yval)):
      I = np.where(y==yval[i])[0]
      count, x0e, x1e = np.histogram2d(X[I,0],X[I,1],[x0e,x1e])
      x0, x1 = np.meshgrid(x0val,x1val)
      plt.scatter(x0.ravel(), x1.ravel(), s=2*count.ravel(),alpha=0.5,
                  c=color[i],edgecolors='none')
  plt.ylim([0,14])
  plt.legend(['benign','malign'], loc='upper right')
  plt.xlabel(xnames[0], fontsize=16)
  plt.ylabel(xnames[1], fontsize=16)
  return plt


# MANUALLY CREATING MODEL BY SETTING BOUNDARY ------

def boundary():
  # set boundary (adjust to see how it changes accuracy)
  bound = 4
  # simple boundary prediction for y
  size = X[:, 0]
  marg = X[:, 1]
  # get vector of y where index of data is shown as true or false depending on if index[i] has (size+marg)>=bound
  # this is the prediction and it creates a line because:
  # marg + size = bound <=> marg = bound - size
  y_pred = (size + marg) >= bound

  # calculate TP, TN, FP and FN rates
  # make vector with each occurence of TP, TN, FP, FN and sum 
  TP = np.sum((y + y_pred) == 2) # y = 1, yhat = 1
  TN = np.sum((y + y_pred) == 0) # y = 0, yhat = 0
  FP = np.sum((y - y_pred) == -1) # y = 0, yhat = 1
  FN = np.sum((y - y_pred) == 1) # y = 1, yhat = 0

  # calculate the following metrics
  accuracy = (TP + TN)/(TP + TN + FP + FN)
  sensitivity = TP/(TP+FN)
  precision = TP/(FP+TP)

  print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
  print("Accuracy is {}, Sensitivity is {}, Precision is {}".format(accuracy, sensitivity, precision))

  plot_weighted_scatter()

# USE LOGISTIC REGRESSION FROM SKLEARN -----------

def logreg():
  clf = LogisticRegression(random_state=0)
  
  # fit the logistic regression model
  clf.fit(X, y)
  
  # predict using the model
  y_pred_clf = clf.predict(X)

  # this is the predicted y with the highest accuracy, but if you want to reduce the false negatives, despite lower accuracy, use probability of class to adjust predictions based on how you want the classes to be weighted
  prob = clf.predict_proba(X)
  print(prob>=0.6)

  y_hat = y_pred_clf

  # Evaluate sklearn's model according to the metrics above
  # calculate TP, TN, FP and FN rates
  # make vector with each occurence of TP, TN, FP, FN and sum 
  TP = np.sum((y + y_hat) == 2) # y = 1, yhat = 1
  TN = np.sum((y + y_hat) == 0) # y = 0, yhat = 0
  FP = np.sum((y - y_hat) == -1) # y = 0, yhat = 1
  FN = np.sum((y - y_hat) == 1) # y = 1, yhat = 0

  # calculate the following metrics using TP, TN, FP, and FN
  accuracy = (TP + TN)/(TP + TN + FP + FN)
  sensitivity = TP/(TP+FN)
  precision = TP/(FP+TP)

  print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
  print("Accuracy is {}, Sensitivity is {}, Precision is {}".format(accuracy, sensitivity, precision))

  # plot boundary line by getting values based on weights
  b = clf.intercept_[0]
  print(b)
  w1, w2 = clf.coef_.T
  #intercept
  c = -b/w2
  # weights are for x and y, so if you divide them it creates a slope
  m = -w1/w2
  print(c,m)
  xd = np.array([0,11])
  yd = m*xd + c
  plt=plot_weighted_scatter()
  plt.plot(xd,yd,"k", lw=1, ls="--")


# logreg()
boundary()
plt.show()
