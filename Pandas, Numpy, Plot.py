import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#running in google colab works much better

# dataframe - df
df = pd.read_csv('https://github.com/ajn313/NYU2023SummerML3/raw/main/Day2/NYU23SummerML1_movie.csv')
# df = df.dropna() to eliminate any null

df.head(20) # get top 20 values from df

#get columns according to headline
names = df['Movie'].values
score = df['IMDB Rating'].values
genre = df['Category'].values

#get each unique genre, and how many times it appears
modes, counts = np.unique(genre, return_counts=True)

#get highest and lowest score using numpy
min_score = np.min(score)
max_score = np.max(score)

#create an array with numpy between the lowest and highest score with increments of 0.5
bin_range = np.arange(min_score, max_score, 0.5)

#get average score
mean_score = np.mean(score)

#get variance of score
var_score = np.var(score)

print("Mean score: ", mean_score)
print("Var of score: ", var_score)

# plot each genre and how many times it appeared with bar graph
plt.bar(modes, counts)
plt.xticks(modes, rotation=90)

#label the sides
plt.xlabel('Score')
plt.ylabel('Number')

plt.show()
