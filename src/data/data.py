import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats
from scipy.stats import norm
from matplotlib import gridspec
import statistics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import requests
import io
import pandas_profiling as pp


import warnings
warnings.filterwarnings("ignore")


# read .csv data
red_wine = pd.read_csv('../wine_data/winequality-red.csv',sep=';')
white_wine = pd.read_csv('../wine_data/winequality-white.csv', sep=';')



#Create Wine category column for red = 0 & white = 1
red_wine['Red0_White1'] = 0

# shift column 'Name' to first position
last_column = red_wine.pop('quality')
  
# insert column using insert(position,column_name,first_column) function
red_wine.insert(12, 'quality', last_column)


#Create Wine category column for red = 0 & white = 1
white_wine['Red0_White1'] = 1

# shift column 'Name' to first position
last_column = white_wine.pop('quality')
  
# insert column using insert(position,column_name,first_column) function
white_wine.insert(12, 'quality', last_column)


# Combine Dataframes together
frames = [red_wine, white_wine]

wine_df = pd.concat(frames).reset_index()
print(wine_df.shape)
wine_df.sample(5)



