from cProfile import run
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

def run_models(model_name):
    print(X.shape)
    bM = bModel(model_name, X, y)
    #bM.train_base_model()
    metrics = bM.train_model()
    with open('metrics.csv', 'a') as f:
        #f.write('Model, training_time, testing_time, train_size, test_size, accuracy, precison, f1, recall')
        #f.write('\n')
        f.write(model_name + '_baseline, '+ metrics)
        f.write('\n')


    for nshard in [5]:
        sM = sModel(model_name, df, nshard)
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
            
        metrics = sM.get_averagedmodel()
    '''
        with open('metrics.csv', 'a') as f:
            #f.write('Model, training_time, testing_time, train_size, test_size, accuracy, precison, f1, recall')
            #f.write('\n')
            f.write(metrics)
            f.write('\n')
    '''



feature_names = ['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products','mean_trnx_amt']
X = df[['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products','mean_trnx_amt','Churn_risk']]
ordered_satisfaction = ['Low', 'Medium', 'High']
X['Churn_risk'] = X.Churn_risk.astype("category").cat.codes
y = np.ravel(X[['Churn_risk']])
X = X.drop(columns=['Churn_risk'])

'''
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
'''


X = df[['Age','Tenure','PSYTE_Segment','Total_score',\
    'Trnx_count','num_products','mean_trnx_amt','Churn_risk']][df.PSYTE_Segment != 12]
ordered_satisfaction = ['Low', 'Medium', 'High']
X['Churn_risk'] = X.Churn_risk.astype("category").cat.codes
y = np.ravel(X[['Churn_risk']])
X = X.drop(columns=['Churn_risk'])


run_models('Mdel 1 ')


X = df[['Age','Tenure','PSYTE_Segment','Total_score',\
    'Trnx_count','num_products','mean_trnx_amt','Churn_risk']][df.Age < 60]
ordered_satisfaction = ['Low', 'Medium', 'High']
X['Churn_risk'] = X.Churn_risk.astype("category").cat.codes
y = np.ravel(X[['Churn_risk']])
X = X.drop(columns=['Churn_risk'])

run_models('Model 2 ')




############
'''
X = df[['Age','Tenure','PSYTE_Segment','Total_score',\
    'Trnx_count','num_products','mean_trnx_amt','Churn_risk']][(df.Total_score > 5)]
ordered_satisfaction = ['Low', 'Medium', 'High']
X['Churn_risk'] = X.Churn_risk.astype("category").cat.codes
y = np.ravel(X[['Churn_risk']])
X = X.drop(columns=['Churn_risk'])


#X = X[X.PSYTE_Segment != 12]

print(X.shape)
bM = bModel(X, y)
#bM.train_base_model()
metrics = bM.train_model()
#metrics = bM.opt_model()

with open('metrics.csv', 'a') as f:
    #f.write('Model, training_time, testing_time, train_size, test_size, accuracy, precison, f1, recall')
    #f.write('\n')
    f.write('Baseline model3,' + metrics)
    f.write('\n')

''''''
X = df[['Age','Tenure','PSYTE_Segment','Total_score',\
    'Trnx_count','num_products','mean_trnx_amt','Churn_risk']][(df.Total_score < 20) | (df.Total_score > 25)]    
ordered_satisfaction = ['Low', 'Medium', 'High']
X['Churn_risk'] = X.Churn_risk.astype("category").cat.codes
y = np.ravel(X[['Churn_risk']])
X = X.drop(columns=['Churn_risk'])


#X = X[X.PSYTE_Segment != 12]

print(X.shape)
bM = bModel(X, y)
#bM.train_base_model()
metrics = bM.train_model()
#metrics = bM.opt_model()

with open('metrics.csv', 'a') as f:
    #f.write('Model, training_time, testing_time, train_size, test_size, accuracy, precison, f1, recall')
    #f.write('\n')
    f.write('Baseline model4,' + metrics)
    f.write('\n')


for nshard in [5]:
    sM = sModel(df, nshard)
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
        metrics = sM.get_averagedmodel()

    with open('metrics.csv', 'a') as f:
        #f.write('Model, training_time, testing_time, train_size, test_size, accuracy, precison, f1, recall')
        #f.write('\n')
        f.write(metrics)
        f.write('\n')


'''