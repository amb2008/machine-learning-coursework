import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#running in google colab works much better
#call desired function at end

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

#print top 5 rows of df
print(df.head(5))

#print entire df shape
print(df.shape)

# set x and y vectors to columns of number of rooms and price
y = df['PRICE'].values
x = df['RM'].values

# scatter plot of rooms in relation to price
plt.plot(x,y,'o')
plt.xlabel('Rooms')
plt.ylabel('Price')
plt.grid()

# create 100 points equally spaced between 3 and 9
xplt = np.linspace(3,9,100)

# create 3 different lines in relation to x
yplt = 9* xplt -30
yplt1 = 12*xplt -53
yplt2 = xplt*13 -60

# just to see how the guess lines stack up, plot them
def guessLines():
  plt.plot(xplt,yplt,'-',linewidth=3, label="model 1")  # Plot the line
  plt.plot(xplt,yplt1,'-',linewidth=3, label="model 2")  # Plot the line
  plt.plot(xplt,yplt2,'-',linewidth=3, label="model 3")  # Plot the line
  plt.xlabel('Average number of rooms in a region')
  plt.ylabel('Price')
  plt.grid()
  plt.legend()

# make vector lines with every x value
yPred1 = 9 *x -30
yPred2 = 12*x -53
yPred3 = 13*x -60

#Error rates
#calculate MAE by squaring vectors realY - predictedY and getting avg
y1SQerr = np.mean((y-yPred1)**2)
y2SQerr = np.mean((y-yPred2)**2)
y3SQerr = np.mean((y-yPred3)**2)

#calculate MAE by getting abs val of vectors realY - predictedY and getting avg
y1ABSerr = np.mean(np.abs(y-yPred1))
y2ABSerr = np.mean(np.abs(y-yPred2))
y3ABSerr = np.mean(np.abs(y-yPred3))

print(y1SQerr, y2SQerr, y3SQerr)
print(y1ABSerr, y2ABSerr, y3ABSerr)

# make vector of ones with number of columns in x, and rows to 1
ones_v = np.ones((x.shape[0], 1))

# make X into a matrix where the first column is 1s and the second is the Xs by horizontally stacking, so the matrix can be multiplied by weights to get w0 + w1x
X = np.hstack([ones_v, x.reshape((-1, 1))])

# calculate weights with formula, creates [w0, w1]
# formula = the (inverse of transposeX * X) * (transposeX * y)
w = (np.linalg.inv(np.transpose(X)@X))@(np.transpose(X))@y
print(w)

# calculate the best fit line by multiplying X matrix with weights
yPred4 = X@(np.transpose(w))

#plot best fit line on top of scatter plot
def predLine():
  plt.plot(x, y, "o", label="train data")
  plt.plot(x, yPred4, "o",label="calculated line")
  plt.legend()

#calculate MSE and MAE best fit line
MSE = np.mean((y-yPred4)**2)
MAE = np.mean(np.abs(y-yPred4))
print(MSE, MAE)

# guessLines()
predLine()
plt.show()
