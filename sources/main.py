import numpy as np
import pandas as pd
from pandas import read_csv
from pyparsing import col
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import classification_report

import pickle

#from sympy import ordered
from lib import *
from data_exploratory_test import *

from originalModel import oModel
from baselineModel import bModel
from shardedModel import sModel

import timeit


#%% Data visualization

# dataset path
#path = 'F:\Machine_Unlearning\Datatest'
path = '../data'
plots_path = '../plots/'

# import dataset
d_name = 'churnsimulateddata.csv'
d = os.path.join(path, d_name)

# read data
df = read_csv(d)
df = df.sample(frac = 1)

# read data
feature_names = ['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products','mean_trnx_amt']


# boxplot
#boxplot_graph(df, feature_names)

# distribution
#dist_graph(df, feature_names)

# coefficient heatmap
#coefficient(df, feature_names)

# 
#hist_graph(df, feature_names)


#%% 
#print(df.describe())
#print(df.columns)


feature_names = ['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products','mean_trnx_amt']
X = df[['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products','mean_trnx_amt','Churn_risk']]
X = X.loc[X['Age'] > np.percentile(X['Age'], 0.05)]

X['Age2'] = np.log10(df['Age'])
X['PSYTE_Segment2'] = np.log10(df['PSYTE_Segment'])
X['Trnx_count2'] = np.log10(df['Trnx_count'])

#X['Tenure'][X['Tenure'] <= 0 ] = 0.001
#X['Trnx_count'][X['Trnx_count'] == 0 ] = 0.001

#X['Tenure2'] = np.log10(df['Tenure'])

ordered_satisfaction = ['Low', 'Medium', 'High']

X['Churn_risk'] = X.Churn_risk.astype("category").cat.codes

#print(X)

#.astype("category", ordered=True, categories=ordered_satisfaction).cat.codes

#print(df.Churn_risk)
y = np.ravel(X[['Churn_risk']])
X = X.drop(columns=['Churn_risk'])


#starttime = timeit.default_timer()
#print("The start time is :",starttime)
print(X.shape)
oM = oModel(X, y)
#bM.train_base_model()
metrics = oM.train_model()
with open('metrics.csv', 'a') as f:
    f.write('Model, training_time, testing_time,  train_size, test_size, accuracy, precison, f1, recall')
    f.write('\n')
    f.write('Original model,' + metrics)
    f.write('\n')

#print(metrics)
#train_model( X, y)



X = df[['Age','Tenure','PSYTE_Segment','Total_score',\
    'Trnx_count','num_products','mean_trnx_amt','Churn_risk']][df.PSYTE_Segment != 12]
X = X.loc[X['Age'] > np.percentile(X['Age'], 0.05)]
X['Age2'] = np.log10(df['Age'])
X['PSYTE_Segment2'] = np.log10(df['PSYTE_Segment'])
X['Trnx_count2'] = np.log10(df['Trnx_count'])
ordered_satisfaction = ['Low', 'Medium', 'High']
X['Churn_risk'] = X.Churn_risk.astype("category").cat.codes
y = np.ravel(X[['Churn_risk']])
X = X.drop(columns=['Churn_risk'])


#X = X[X.PSYTE_Segment != 12]

print(X.shape)
bM = bModel(X, y)
#bM.train_base_model()
metrics = bM.train_model()
with open('metrics.csv', 'a') as f:
    #f.write('Model, training_time, testing_time, train_size, test_size, accuracy, precison, f1, recall')
    #f.write('\n')
    f.write('Baseline model1,' + metrics)
    f.write('\n')


X = df[['Age','Tenure','PSYTE_Segment','Total_score',\
    'Trnx_count','num_products','mean_trnx_amt','Churn_risk']][df.Age < 60]
X = X.loc[X['Age'] > np.percentile(X['Age'], 0.05)]
X['Age2'] = np.log10(df['Age'])
X['PSYTE_Segment2'] = np.log10(df['PSYTE_Segment'])
X['Trnx_count2'] = np.log10(df['Trnx_count'])
ordered_satisfaction = ['Low', 'Medium', 'High']
X['Churn_risk'] = X.Churn_risk.astype("category").cat.codes
y = np.ravel(X[['Churn_risk']])
X = X.drop(columns=['Churn_risk'])


#X = X[X.PSYTE_Segment != 12]

print(X.shape)
bM = bModel(X, y)
#bM.train_base_model()
metrics = bM.train_model()
with open('metrics.csv', 'a') as f:
    #f.write('Model, training_time, testing_time, train_size, test_size, accuracy, precison, f1, recall')
    #f.write('\n')
    f.write('Baseline model2,' + metrics)
    f.write('\n')


sM = sModel(df)
metrics = sM.train_model()
with open('metrics.csv', 'a') as f:
    #f.write('Model, training_time, testing_time, train_size, test_size, accuracy, precison, f1, recall')
    #f.write('\n')
    f.write(metrics)
    f.write('\n')

metrics = sM.get_aggregatedmodel()
with open('metrics.csv', 'a') as f:
    #f.write('Model, training_time, testing_time, train_size, test_size, accuracy, precison, f1, recall')
    #f.write('\n')
    f.write(metrics)
    f.write('\n')



