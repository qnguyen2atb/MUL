import numpy as np
import pandas as pd
from pyparsing import col
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import classification_report

import pickle

#from sympy import ordered
from lib import *
from data_exploratory_test import *

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
feature_names = ['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products','mean_trnx_amt']

# boxplot
boxplot_graph(df, feature_names)

# distribution
dist_graph(df, feature_names)

# coefficient heatmap
coefficient(df, feature_names)

# 
hist_graph(df, feature_names)

#%% 
#print(df.describe())
#print(df.columns)



#df = df.dropna()
X = df[['Age','Tenure','PSYTE_Segment','Total_score','Trnx_count','num_products','mean_trnx_amt','Churn_risk']]
#print(df_b.isna().sum())
#print(df_b.describe())

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

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
#print(X_train)
#print(y_train)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100,  n_jobs=-1,) #100

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)
precison = metrics.precision_score(y_test, y_pred, average='weighted')
print('Precison: ', precison)
recall = metrics.recall_score(y_test, y_pred, average='weighted')
print('Recall: ', recall)
f1 = metrics.f1_score(y_test, y_pred, average='weighted')
print('F1: ', f1)
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)


#plot feature important
#Check important feature
# check Important features
feature_importances_df = pd.DataFrame(
    {"feature": list(X.columns), "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)

# Display
print(feature_importances_df)
# visualize important featuers
# Creating a bar plot
fig = plt.figure(figsize=[15,15])
ax = fig.add_axes([0.1,0.3,0.8,0.65])
ax.bar(x=feature_importances_df.feature, height=feature_importances_df.importance)
# Add labels 
plt.ylabel("Feature Importance Score",fontsize=20)
plt.xlabel("Features",fontsize=20)
#plt.title("Visualizing Important Features",fontsize=20)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
#    , horizontalalignment="right", fontweight="light", fontsize="x-large")
plotname='feature_important'
plt.savefig('../plots/'+plotname+'.png')
plt.show()




# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(clf, open('../models/'+filename, 'wb'))

 
# load the model from disk
loaded_model = pickle.load(open('../models/'+filename, 'rb'))
#result = loaded_model.score(X_test, y_test)

#classification_report(X_test, y_test)

