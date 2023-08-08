import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# create two linear models, one lasso, one ridge (both prevent overfitting), and see which one has a better error rate
reg1 = linear_model.Lasso(alpha=.02, fit_intercept=False, max_iter=10000)
reg2 = linear_model.Ridge(alpha=.05, fit_intercept=False, max_iter=10000)

# create dataframe of x axis fish features 
featureurl = 'https://raw.githubusercontent.com/ajn313/NYU2023SummerML3/main/Day5/fish_market_feature.csv'
featuredf = pd.read_csv(featureurl)

# create dataframe of fish weights
yurl = 'https://raw.githubusercontent.com/ajn313/NYU2023SummerML3/main/Day5/fish_market_label.csv'
ydf = pd.read_csv(yurl)
y = ydf['Weight'].values

# isolate each feature
l1 = featuredf['Length1'].values
l2 = featuredf['Length2'].values
l3 = featuredf['Length3'].values
h = featuredf['Height'].values
wi = featuredf['Width'].values

# plot each feature in correlation with weight
def plotFeatureCor():
  plt.plot(l1, y, "o", label='Length 1')
  plt.plot(l2, y, "o", label='Length 2')
  plt.plot(l3, y, "o", label='Length 3')
  plt.plot(h, y, "o", label='Height')
  plt.plot(wi, y, "o", label='Width')
  plt.legend()
  plt.xlabel('Feature Measurement')
  plt.ylabel('Fish Weight')

# create a vector of ones for the design matrix

def design_matrix(x1, x2, x3, x4, x5, M):
    # reshape each feature to be many rows by one column
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    x3 = x3.reshape(-1, 1)
    x4 = x4.reshape(-1, 1)
    x5 = x5.reshape(-1, 1)
    ones = np.ones((x1.shape[0], 1))
    X = ones
    for i in range(M):
      #make a new vector x to the power of i until m
      #stack each vector to create matrix where each row looks like:
      #[1, x^1, x^2, ..., x^M]
      newX1 = x1**(i+1)
      newX2 = x2**(i+1)
      newX3 = x3**(i+1)
      newX4 = x4**(i+1)
      newX5 = x5**(i+1)
      X = np.hstack([X, newX1, newX2, newX3, newX4, newX5])
    return X


# FIND BEST VALUE OF M ----

mse_s = np.zeros((30, 1))

def findBestM():
  for M in range(30):
    X = design_matrix(l1, l2, l3, h, wi, M)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=42)
    reg2.fit(X_train, y_train)
    y_pred_new = reg2.predict(X_val)
    mse_s[M] = np.mean((y_val-y_pred_new)**2)
  
  min = np.argmin(mse_s)
  print("Best value of M (degree for matrix): ", min+1)

# create the design matrix with M = 4
X = design_matrix(l1, l2, l3, h, wi, 4)


# TRAIN MODELS -------


# split X into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=42)

# train the models on only the training data
reg1.fit(X_train, y_train.reshape((-1, 1)))
reg2.fit(X_train, y_train.reshape((-1, 1)))

# predict the training outputs
y_train1 = reg1.predict(X_train)
y_train2 = reg2.predict(X_train)

# find the training error rate
RMSET1 = np.sqrt(np.mean((y_train.reshape(-1,1) -y_train1)**2) )
RMSET2 = np.sqrt(np.mean((y_train.reshape(-1,1) -y_train2)**2) )
print(f"The train error is lasso, {RMSET1:.3f}")
print(f"The train error is ridge, {RMSET2:.3f}")

# predict the validation output
yval1 = reg1.predict(X_val)
yval2 = reg2.predict(X_val)

# calculate the validation error rate
RMSEv1 = np.sqrt(np.mean((y_val.reshape(-1,1) -yval1)**2) )
RMSEv2 = np.sqrt(np.mean((y_val.reshape(-1,1) -yval2)**2) )
print(f"The val error is lasso, {RMSEv1:.3f}")
print(f"The val error is ridge, {RMSEv2:.3f}")
# Ridge is better


# USE THE MODEL ON TESTING DATA ------------


# create an x dataframe
testXurl = 'https://raw.githubusercontent.com/ajn313/NYU2023SummerML3/main/Day5/fish_market_test_feature.csv'
testXdf = pd.read_csv(testXurl)

#isolate the features
l1 = testXdf.iloc[:, 0].values
l2 = testXdf.iloc[:, 1].values
l3 = testXdf.iloc[:, 2].values
h = testXdf.iloc[:, 3].values
wi = testXdf.iloc[:, 4].values

# create a y dataframe
testYurl = 'https://raw.githubusercontent.com/ajn313/NYU2023SummerML3/main/Day5/fish_market_test_label.csv'
testYdf = pd.read_csv(testYurl)

y = testYdf.iloc[:, 0].values

# create the design matrix
X = design_matrix(l1, l2, l3, h, wi, 4)

# predict y
y_pred = reg2.predict(X)
y_pred = y_pred.reshape(-1, 1)

# find error rate
RMSE = np.sqrt(np.mean((y.reshape(-1,1) - y_pred)**2))
print(RMSE)

# since the y prediction vector has 30 values, create an evenly spaced 30 values to plot each prediction.
xplt = np.linspace(0, 60, 30)

# plot real y in comparison to predicted y
plt.xlabel('Index of prediction * 2')
plt.ylabel('Fish Weight')
plt.plot(xplt, y, "o", label = "Actual Y")
plt.plot(xplt, y_pred, "o", label = "Predicted Y")
plt.legend()

reg1 = linear_model.Ridge(alpha=.05, fit_intercept=False, max_iter=10000)

plt.show()
