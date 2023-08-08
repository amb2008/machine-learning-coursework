import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#running in google colab works much better

#list of header names
names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','PRICE']

#make dataframe from csv url
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                header=None, names=names , delim_whitespace = True, na_values='?')

"""
Attribute Information:
    1.  CRIM      per capita crime rate by town
    2.  ZN        proportion of residential land zoned for lots over
                  25,000 sq.ft.
    3.  INDUS     proportion of non-retail business acres per town
    4.  CHAS      Charles River dummy variable (= 1 if tract bounds
                  river; 0 otherwise)
    5.  NOX       nitric oxides concentration (parts per 10 million)
    6.  RM        average number of rooms per dwelling
    7.  AGE       proportion of owner-occupied units built prior to 1940
    8.  DIS       weighted distances to five Boston employment centres
    9.  RAD       index of accessibility to radial highways
    10. TAX       full-value property-tax rate per $10,000
    11. PTRATIO   pupil-teacher ratio by town
    12. B         1000(Bk - 0.63)^2 where Bk is the proportion of blocks by town
    13. LSTAT     % lower status of the population
    14. MEDV      Median value of owner-occupied homes in $1000's
"""

#remove price header, because that will be set seperately as y
features=df.columns.to_list()
features.remove('PRICE')
print(features)

# Make X matrix into features
X = df[features].values

# Make Y into vector of prices 
y = df['PRICE'].values
print(y.shape)
y = y.reshape(-1,1)



# USING SCIKIT LINEAR REGRESSION MODEL ----------------------


# get linear regression model from scikit, optimize, give it X and y
regr = LinearRegression(fit_intercept=True)
regr.fit(X, y)

# create a y vector of predicted values based on trained linear regression model and X matrix
y_hat = regr.predict(X)  # Model prediction
print(y_hat.shape)

# print the weights
print(regr.coef_)        # this is [w_1, ...., w_n] the weights

#print the bias
print(regr.intercept_)   # this the bias w_0 

#create matrix of real y and predicted y, so they can be compared
Y = np.hstack([y, y_hat])
with np.printoptions(precision=2): #precision is how many decimal points
    print(Y[:10,:]) #print first ten rows



# CREATING LINEAR REGRESSION MODEL MANUALLY



# create a vector of ones that will fit with X matrix
ones_v = np.ones((X.shape[0], 1))

#stack ones with X matrix so it can multiplied with weights correctly
OX = np.hstack([ones_v, X])

#calculate weights with formula
w = np.linalg.inv(np.transpose(OX)@OX)@(np.transpose(OX))@y
print(np.transpose(w))

#calculate predicted y with out calculated weights
y_hat = OX@w

# create an array with every value from 0 to the number of columns in y
# so that the graph shows Y at every x
xplt = np.arange(y.shape[0])
plt.figure(figsize=(17,7))
plt.plot(xplt, y, "o", label="actual")
plt.plot(xplt, y_hat, "o", label="predicted")
plt.legend()

plt.show()

# to plot just a few
# xplt = np.arange(10)
# plt.figure(figsize=(17,7))
# plt.plot(xplt, y[:10], "o", label="actual")
# plt.plot(xplt, y_hat[:10], "o", label="predicted")
# plt.legend()
