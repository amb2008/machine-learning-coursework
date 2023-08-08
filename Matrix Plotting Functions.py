#when using with replit, plot will show in output window
#running in google colab works much better
import numpy as np
import matplotlib.pyplot as plt

#call functions at the end

#scatter plot 4 points using vectors
def plot1():
  x1 = [0, 3, 5, 4]
  y1 = [1, 3, 2, 1]
  #plotted as (x1[i], y1[i])
  plt.scatter(x1, y1)
  plt.xlabel('time')
  plt.ylabel('score')
  plt.grid()

#plot fours points using vectors connected by lines
def plot2():
  plt.show()
  x1 = [0, 5, 3, 4]
  y1 = [1, 2, 3, 1]
  #lines connecting (x1[i], y1[i])

  fig = plt.figure(figsize=(15, 5))
  plt.plot(x1, y1, linewidth=4)
  # replace with following to make into scatter plot: plt.plot(x1, y1, "o")
  plt.xlabel('x')
  plt.ylabel('y')

# plot line by uneccesarrily creating many points
def plot3():
  x = np.linspace(-1, 5, 100)
  #creates x with each being equally spaced from eachother between -1 and 5
  y2 = 2 * x + 0.5
  #create y corresponding with x for a linear function
  plt.plot(x, y2)
  plt.xlabel('x')
  plt.ylabel('y')

# plot quadratic by assigning y in relation to x
def plot4():
  x = np.linspace(-5, 5, 100)
  # create 100 values equally spaced between -5, 5
  y = x**2
  # y = x^2 for a quadratic.
  plt.plot(x, y)
  plt.xlabel('x')
  plt.ylabel('y')


x = np.linspace(-5, 5, 1000)

# plot cubic using power function in relation to x
def plot5():
  np.pi #pi
  poly = np.power(x, 3) + 2
  # making y = x^3
  plt.plot(x, poly)

# plot exponential function
def plot6():
  exp = np.exp(x)
  plt.plot(x, exp)


plot2()
plt.show()
