import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# get csv
url = 'https://raw.githubusercontent.com/ajn313/NYU2023SummerML3/main/Day3/polyfit_data.csv'

# read csv
df = pd.read_csv(url)

# get csv x and y columns
x = df['x'].values
y = df['y'].values

# plot values
plt.plot(x,y,'o')
plt.xlabel('x')
plt.ylabel('y')

# Choose integer value for M, and find which one has the least error (MSE)
M = 5

# make ones vector and set it as the start to the design matrix X
ones = np.ones((df.shape[0], 1))
X = ones

# Make a polynomial fitted design matrix
def design_matrix(x, M):
    x = x.reshape(-1, 1)
    X = ones
    for i in range(M):
      #make a new vector x to the power of i until m
      #stack each vector to create matrix where each row looks like:
      #[1, x^1, x^2, ..., x^M]
      newX = x**(i+1)
      X = np.hstack([X, newX])
    return X

# compute the design matrix of x
X = design_matrix(x, M)

# Reshape y to a column vector
y = y.reshape(-1, 1)


# MANUALLY CREATE MODEL ---------------


# compute w vector using the least-square solution
w = np.linalg.pinv(X)@y

# find y prediction and compute the MSE (error rate)
y_hat = X@w
MSE = np.mean((y-y_hat)**2)
print("Mean squared error with M: ", M)
print(MSE, "\n")

# plot actual y and predicted y
plt.plot(x, y, "o", label="actual")
plt.plot(x, y_hat, "o", label="prediction")
plt.legend()

# plot trend line
xplt = np.linspace(0, 5, 20)
XPLT = design_matrix(xplt, M)
yplt = XPLT@w
plt.plot(xplt, yplt)
plt.grid()


# USING SCIKIT MODEL ---------------


# create a regression model
reg = linear_model.LinearRegression(fit_intercept=False)

# fit the model with the design matrix and y
reg.fit(X, y)

# use the model to predict y vector
yhat = reg.predict(X)

# plot the model with scikit, same as manual plot
plt.plot(x, y, 'o')
plt.plot(x, yhat, 'o')


# FIND BEST VALUE FOR M -------------


# create a set of M values to test
Ms_test = np.arange(1, 21)
mse_s = np.zeros((Ms_test.shape[0], 1))

# for every power 1 to 20:
for i, M in enumerate(Ms_test):
    # reshape x into column vector
    x = x.reshape(-1, 1)
    X = ones
    # compute Design matrix
    for i in range(M):
      newX = x**(i+1)
      X = np.hstack([X, newX])

    # compute least-square solution (w)
    w = np.linalg.pinv(X)@y

    # compute the mse on the predicted data and store it:
    mse_s[i] = np.mean((y-X@w)**2)


# printing index of minimum value in MSE vector
# index + 1 will be the best M value
min = np.argmin(mse_s)
print("Best value of M (degree for matrix): ", min+1)

plt.show()

