#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:55:14 2022

@author: quang
"""
from lib import *

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline


class bModel():
    
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3) 

    def train_model(self):
        
        original_params = {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}

        clf=RandomForestClassifier(**original_params) #100
        
        pipe_rf = Pipeline([('clf', RandomForestClassifier(random_state=42, verbose=0))])
        
        # Set grid search params
        param_range = range(10,20)
        param_estimators = range(100,500,20)
        param_min_leaf = [0.1,0.2,0.3,0.4,0.5]
        param_min_weight = [0, 0.5]
        param_max_features = ['auto', 'sqrt', 'log2']
        param_max_nodes = [None, 2,3,4]
        param_range_fl = [1.0, 0.1]
        
        #test set
        param_estimators = [100]
        param_range = [10,15,20]
        param_min_leaf = [0.1]
        param_min_weight = [0]
        param_max_features = ['auto']
        param_max_nodes = [None]
        param_range_fl = [1.0]
        
        
        grid_params_rf = [{'clf__n_estimators': param_estimators, 
                           'clf__max_depth': param_range, 
                           'clf__min_samples_split': param_range_fl,
                           'clf__criterion': ['gini', 'entropy'],
                           'clf__min_samples_leaf': param_min_leaf, 
                           'clf__min_weight_fraction_leaf': param_min_weight, 
                           'clf__max_features': param_max_features, 
                           'clf__max_leaf_nodes': param_max_nodes,
                           'clf__min_impurity_decrease': param_range_fl, 
                           #'clf__min_impurity_split': range(1,10),  depreciated 
                         }]
        jobs = -1
        
        RF = GridSearchCV(estimator=pipe_rf,
                param_grid=grid_params_rf,
                scoring='accuracy',
                cv=10, 
                n_jobs=jobs)
        
        #Train the model using the training sets y_pred=clf.predict(X_test)
        RF.fit(self.X_train,self.y_train)
        
        print('Best params are : %s' % RF.best_params_)
        # Best training data accuracy
        print('Best training accuracy: %.3f' % RF.best_score_)
        # Predict on test data with best params
        self.y_pred = RF.predict(self.X_test)
        # Test data accuracy of model with best params
        print('Test set accuracy score for best params: %.3f ' % accuracy_score(self.y_test, self.y_pred))
        # Track best (highest test accuracy) model
        if accuracy_score(self.y_test, self.y_pred) > RF.best_score_:
            best_acc = accuracy_score(self.y_test, self.y_pred)
        
        
        clf=RandomForestClassifier(RF.best_params_) #100
        clf.fit(self.X_train,self.y_train)
        
        
        '''    
        self.y_pred=clf.predict(self.X_test)
        precison = metrics.precision_score(self.y_test, self.y_pred, average='weighted')
        print('Precison: ', precison)
        recall = metrics.recall_score(self.y_test, self.y_pred, average='weighted')
        print('Recall: ', recall)
        f1 = metrics.f1_score(self.y_test, self.y_pred, average='weighted')
        print('F1: ', f1)
        accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        print('Accuracy: ', accuracy)
        
        '''
        #plot feature important
        #Check important feature
        # check Important features
        feature_importances_df = pd.DataFrame(
            {"feature": list(self.X_train.columns), "importance": clf.feature_importances_}
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
        filename = 'original_model.sav'
        pickle.dump(clf, open('../models/'+filename, 'wb'))
        
        # load the model from disk
        loaded_model = pickle.load(open('../models/'+filename, 'rb'))
        #result = loaded_model.score(X_test, y_test)
    
    
    
    
    def train_model(self):
    

        clf=RandomForestClassifier(n_estimators=100, max_leaf_nodes=None, n_jobs=-1) #100
        

        starttime = timeit.default_timer()
        #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(self.X_train,self.y_train)
        model_params = clf.get_params() 
        print(model_params)

        training_time = timeit.default_timer() - starttime
        print("The training time is :", training_time)
        starttime = timeit.default_timer()
        self.y_pred=clf.predict(self.X_test)
        precison = metrics.precision_score(self.y_test, self.y_pred, average='weighted')
        print('Precison: ', precison)
        recall = metrics.recall_score(self.y_test, self.y_pred, average='weighted')
        print('Recall: ', recall)
        f1 = metrics.f1_score(self.y_test, self.y_pred, average='weighted')
        print('F1: ', f1)
        accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        print('Accuracy: ', accuracy)
        testing_time = timeit.default_timer() - starttime
        print("The testing time is :", testing_time)
        
        #plot feature important
        #Check important feature
        # check Important features
        feature_importances_df = pd.DataFrame(
            {"feature": list(self.X_test.columns), "importance": clf.feature_importances_}
            ).sort_values("importance", ascending=False)
        
        # Display
        #print(feature_importances_df)
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
        #plt.show()
        
        
        # save the model to disk
        filename = 'original_model.sav'
        pickle.dump(clf, open('../models/'+filename, 'wb'))
        
        # load the model from disk
        loaded_model = pickle.load(open('../models/'+filename, 'rb'))
        #result = loaded_model.score(X_test, y_test)
        return str(str(format(training_time, ".3f")) + ',' +
               str(format(testing_time, ".3f")) + ',' +
               str(format(self.y_train.shape[0], ".0f")) +  ',' +
               str(format(self.y_test.shape[0], ".0f")) + ',' +
               str(format(accuracy, ".3f")) +  ',' +
               str(format(precison, ".3f")) + ',' +
               str(format(f1, ".3f"))+ ',' +
               str(format(recall, ".3f")))
