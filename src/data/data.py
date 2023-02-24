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
#import pandas_profiling as pp


import warnings
warnings.filterwarnings("ignore")


def combine_data(red_wine_data_path: str=None,
                 white_wine_data_path: str=None
                ):

    # read .csv data
    red_wine = pd.read_csv(red_wine_data_path, sep=';')
    white_wine = pd.read_csv(white_wine_data_path, sep=';')



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

        
    return wine_df


def preprocess_df(df, target_col='quality', test_size=0.2, random_state=0):
    """
    Preprocesses a pandas dataframe by standard scaling the features and splitting
    the data into training and test sets using cross validation.

    Parameters:
        df (pd.DataFrame): The input dataframe to be preprocessed.
        target_col (str): The name of the target column in the dataframe.
        test_size (float, optional): The size of the test set as a fraction of the data. Default is 0.2.
        random_state (int, optional): The seed for the random number generator used in train_test_split. Default is 0.

    Returns:
        X_train (np.array): The preprocessed training set features.
        X_test (np.array): The preprocessed test set features.
        y_train (np.array): The training set target values.
        y_test (np.array): The test set target values.
    """
    # Separate the target column from the features
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Save the training and test sets as separate CSV files
    pd.DataFrame(X_train, columns=df.drop(columns=[target_col]).columns).to_csv("train_data/train_features.csv", index=False)
    pd.DataFrame(X_test, columns=df.drop(columns=[target_col]).columns).to_csv("train_data/test_features.csv", index=False)
    pd.DataFrame(y_train, columns=[target_col]).to_csv("train_data/train_target.csv", index=False)
    pd.DataFrame(y_test, columns=[target_col]).to_csv("train_data/test_target.csv", index=False)

    return X_train, X_test, y_train, y_test


