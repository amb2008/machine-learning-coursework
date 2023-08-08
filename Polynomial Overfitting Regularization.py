import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


nsamp = 25 # number of samples taken
p = np.array([5,1,-2,-.5]) # true coefficients
var = 0.1 # noise variance

# make vector of x with 25 values between -1 and 1
x = np.linspace(-1,1,nsamp)

# for every x, multiply by vector p: 
# p[0]x[i]^3 + p[1]x[i]^2 + p[2]x[i] + p[3]
y_true = np.polyval(p,x)

# plot polynomial
plt.plot(x,y_true, label="True Trend")

# create a more scattered, noisy y vector
y = y_true + np.random.normal(0, np.sqrt(var), nsamp)

# we can force a scatter plot in plt.plot by making the third argument 'o'
plt.plot(x,y,'o', label="Data Points");
plt.grid();
plt.xlabel('x')
plt.ylabel('y')

# specifiy split between train, test, and validation split
ntrain = 15
nval = 5
ntest = 5

# randomize the order of the numbers in the x vector
inds = np.random.permutation(nsamp)

# create individual vectors with values for train, validation, and testing
train_choices = inds[:ntrain]
val_choices = inds[ntrain:ntrain+nval]
test_choices = inds[ntrain+nval:]

xtrain, ytrain = x[train_choices], y[train_choices]
xval, yval     = x[val_choices], y[val_choices]
xtest, ytest   = x[test_choices], y[test_choices]

# create a column vector of ones
ones = np.ones((15, 1))

# forming the design matrix
# features x, model order M
def design_matrix(x, M):
    # create the array of ones
    Design_Matrix = np.ones((x.shape[0],M+1)) # use the np.ones function

    # use a for loop to populate the Design_Matrix columnwise
    #stack each vector to create matrix where each row looks like:
    #[1, x^1, x^2, ..., x^M]
    for j in range(M):
        Design_Matrix[:,j+1] = x**(j+1)

    return Design_Matrix

# set M and create design Matrices for each dataset
M = 8
Xtrain = design_matrix(xtrain, M)
Xval = design_matrix(xval,M)
Xtest = design_matrix(xtest,M)


# fit the polynomial model using linear regression
reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit(Xtrain, ytrain.reshape((-1, 1)))
w = reg.coef_


# find and print the training error
# 1. generate the predictions on the training data
# 2. use the skearn metric MSE to calculate RMSE (note: pass squared=False)
yhat = reg.predict(Xtrain)
RMSE = np.sqrt(np.mean((ytrain-yhat)**2) )
print("Train RMSE = %.4f" % RMSE)


# find and print the test error RMS
# 1. generate the predictions on the testing data
# 2. use the skearn metric MSE to calculate RMSE (note: pass squared=False)
Xtest = design_matrix(xtest, M)
yhat = reg.predict(Xtest)
RMSE = np.sqrt( np.mean((ytest-yhat)**2) )
print("Test RMSE = %.4f" % RMSE)

# creating and plotting trend line
x_line = np.linspace(-1,1,500)
X_line = design_matrix(x_line, M)
y_line = reg.predict(X_line)

def plotTrend1():
  plt.figure()
  plt.plot(x_line, y_line)
  plt.plot(xtrain,ytrain,'o',markeredgecolor='black')
  plt.plot(xtest,yval,'o',markeredgecolor='black')
  plt.legend(['Model','Train Points', 'Validation Points'])



mse=mean_squared_error

# Using the following models to fit the data with regularization
# Lasso and Ridge are two methods of regularization
reg1 = linear_model.Lasso(alpha=.02, fit_intercept=False)
reg2 = linear_model.Ridge(alpha=.05, fit_intercept=False)

# Train both models
reg1.fit(Xtrain, ytrain.reshape((-1, 1)))
reg2.fit(Xtrain, ytrain.reshape((-1, 1)))

# Calculate losses for both models under each dataset
mseTr1 = mse(ytrain, reg1.predict(Xtrain), squared=False)
mseTr2 = mse(ytrain, reg2.predict(Xtrain), squared=False)

print(mseTr1, mseTr2)

mseV1 = mse(yval, reg1.predict(Xval), squared=False)
mseV2 = mse(yval, reg2.predict(Xval), squared=False)

print(mseV1, mseV2)

mseTe1 = mse(ytest, reg1.predict(Xtest), squared=False)
mseTe2 = mse(ytest, reg2.predict(Xtest), squared=False)

print(mseTe1, mseTe2)

# create trend lines for each model
x_line = np.linspace(-1,1,200)
X_line = design_matrix(x_line, M)
y_line0 = reg.predict(X_line)
y_line1 = reg1.predict(X_line)
y_line2 = reg2.predict(X_line)

def plotTrends():
  print("plotting trends")
  plt.plot(x_line, y_line0, label="Not regularized")
  plt.plot(x_line, y_line1, label="Lasso")
  plt.plot(x_line, y_line2, label= "Ridge")
  plt.show()

w = reg.coef_
# print out w to see weights if desired
plotTrends()
plt.legend()
plt.show()
