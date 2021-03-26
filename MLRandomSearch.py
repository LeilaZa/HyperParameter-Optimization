#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:54:27 2020
Baseline Machine Learning models: Classifiers with Default Hyperparameters
@author: leila zahedi
"""

#-------------------------------------------------------------------------------------------------------------------------
#-------------------------- Machine Learning models: Classifiers with Randomized Search---------------------------
#-------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------Main Libraries-------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import datetime
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import randint as sp_randint
from numpy import mean
from sklearn.model_selection import KFold
from scipy.stats import randint as sp_randint
from random import randrange as sp_randrange
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
#----------------------------------------------------Split Dataset-------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=101)

# define cv
cv = KFold(n_splits=3, random_state=100, shuffle=True)
#create results file
open('RandomizedSearch', 'w').close()
# number of iterations for random search

#-------------------------------------------------------Decision Tree---------------------------------------------
print("------------------------------------------------")
print("Decision Tree")
print("------------------------------------------------")
from sklearn.tree import DecisionTreeClassifier

rf_params = {
    'max_features': ['sqrt','auto','log2',None],
    'max_depth': sp_randint(5,50),
    'min_samples_leaf': sp_randint(1,15),
    'min_samples_split': sp_randint(2,30),
    'criterion':['gini','entropy'],
    'random_state': [100]
    } #121000
n_iter_search=100
# for k in rf_params:
#     n_iter_search=round(n_iter_search*len(rf_params[k]))
# n_iter_search=n_iter_search*0.25

time1 = datetime.datetime.now()# start time
dtree_classifier= DecisionTreeClassifier()
Random = RandomizedSearchCV(dtree_classifier, param_distributions=rf_params,n_iter=n_iter_search,cv=cv,scoring='accuracy')
Random.fit(X, y)
time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(Random.best_params_)
print('Accuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open('results_RandomizedSearch', 'a') as results:
    results.write("\n--------------------------------------------*RandomizedSearch*-----------------------------------------------\n")
    results.write("\n------------------------------------------------\nDecision Tree\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(Random.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")


#------------------------------------------------------Multinomial Naive Bayes-------------------------------------------
print("Naive Bayes")
print("------------------------------------------------")
from sklearn.naive_bayes import MultinomialNB

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

Xnb= NormalizeData(X)
ynb=y.astype("category")
ynb = ynb.cat.codes
ynb = ynb.replace({0:1, 1:0})

rf_params = {
    'alpha': stats.uniform(0.1,1),
    'fit_prior': [True, False],
    #'class_prior':[]
    }

n_iter_search=10
# for k in rf_params:
#     n_iter_search=round(n_iter_search*len(rf_params[k]))
# n_iter_search=n_iter_search*0.25

time1 = datetime.datetime.now()# start time
mNB_classifier = MultinomialNB()
Random = RandomizedSearchCV(mNB_classifier, param_distributions=rf_params,n_iter=n_iter_search,cv=cv,scoring='accuracy')
Random.fit(Xnb, ynb)
time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(Random.best_params_)
print('Accuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open('results_RandomizedSearch', 'a') as results:
    results.write("\n------------------------------------------------\nNaive Bayes\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(Random.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")

#--------------------------------------------------------Support Vector Machine-------------------------------------------
print("Support Vector Machine")
print("------------------------------------------------")
from sklearn.svm import SVC #,SVR

rf_params = {
    'C': stats.uniform(0,50),
    'kernel':['linear','poly','rbf','sigmoid'],
    'random_state': [100]
}

n_iter_search=15
# for k in rf_params:
#     n_iter_search=round(n_iter_search*len(rf_params[k]))
# n_iter_search=n_iter_search*0.25

time1 = datetime.datetime.now()# start time
svm_classifier= SVC()
Random = RandomizedSearchCV(svm_classifier, param_distributions=rf_params,n_iter=n_iter_search,cv=cv,scoring='accuracy')
Random.fit(X, y)
time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(Random.best_params_)
print('Accuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open('results_RandomizedSearch', 'a') as results:
    results.write("\n------------------------------------------------\nSupport Vector Machine\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(Random.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")


#--------------------------------------------------------XGBoost-------------------------------------------
print("XGBoost")
print("------------------------------------------------")
from xgboost import XGBClassifier

rf_params = {
    'n_estimators': sp_randint(5,500),
    'learning_rate': stats.uniform(0,1),
    'max_depth': sp_randint(5,50),
    'subsample' : stats.uniform(0,1),
    'colsample_bytree':stats.uniform(0,1),
}

n_iter_search=300
# for k in rf_params:
#     n_iter_search=round(n_iter_search*len(rf_params[k]))
# n_iter_search=n_iter_search*0.25

y1= y.astype('category')
y1 = y1.cat.codes
#X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size=0.2,random_state=101)

time1 = datetime.datetime.now()# start time
xgb_classifier= XGBClassifier(objective='binary:logistic', use_label_encoder=False, random_state=100)
Random = RandomizedSearchCV(xgb_classifier, param_distributions=rf_params,n_iter=n_iter_search,cv=cv,scoring='accuracy')
Random.fit(X, y1)
time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(Random.best_params_)
print('Accuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open('results_RandomizedSearch', 'a') as results:
    results.write("\n------------------------------------------------\nXGBoost\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(Random.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")

#--------------------------------------------------------K-Nearest Neighbor-------------------------------------------
print("KNN")
print("------------------------------------------------")
from sklearn.neighbors import KNeighborsClassifier #, KNeighborsRegressor

rf_params = {
    'n_neighbors': sp_randint(1,20),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
} #160

n_iter_search=50
# for k in rf_params:
#     n_iter_search=round(n_iter_search*len(rf_params[k]))
# n_iter_search=n_iter_search*0.25

time1 = datetime.datetime.now()# start time
kn_classifier= KNeighborsClassifier()
Random = RandomizedSearchCV(kn_classifier, param_distributions=rf_params,n_iter=n_iter_search,cv=cv,scoring='accuracy')
Random.fit(X, y)
time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(Random.best_params_)
print('Accuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open('results_RandomizedSearch', 'a') as results:
    results.write("\n------------------------------------------------\nK-Nearest Neighbors\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(Random.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")


#--------------------------------------------------------LR-------------------------------------------
print("Logistic Regression_GridSearch")
print("------------------------------------------------")
from sklearn.linear_model import LogisticRegression

# Define the hyperparameter configuration space
rf_params = {
    'penalty' : [ 'l1', 'l2' , 'elasticnet'],
    'C' : stats.uniform(0,1),
    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'random_state': [100]
    #'class_weight': ['balanced', None ]
} #150

n_iter_search=50
# for k in rf_params:
#     n_iter_search=round(n_iter_search*len(rf_params[k]))
# n_iter_search=n_iter_search*0.25

time1 = datetime.datetime.now()# start time
lr_classifier= LogisticRegression(max_iter=10000)
Random = RandomizedSearchCV(lr_classifier, param_distributions=rf_params,n_iter=n_iter_search,cv=cv,scoring='accuracy')
Random.fit(X, y)
time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(Random.best_params_)
print('Accuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open('results_RandomizedSearch', 'a') as results:
    results.write("\n------------------------------------------------\nLogistic Regression\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(Random.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")


#--------------------------------------------------------Random Forest-------------------------------------------
print("Random Forest_RandomizedSearch")
print("------------------------------------------------")
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier

# Define the hyperparameter configuration space
rf_params = {
    'n_estimators': sp_randint(5,500),
    'max_features': ['sqrt','auto','log2'],
    'max_depth': sp_randint(5,50),
    'min_samples_leaf': sp_randint(1,15),
    'min_samples_split':sp_randint(2,30),
    'criterion':['gini','entropy'],
    'random_state': [100]
} 

n_iter_search=300
# for k in rf_params:
#     n_iter_search=round(n_iter_search*len(rf_params[k]))
# n_iter_search=n_iter_search*0.25

time1 = datetime.datetime.now()# start time
rf_classifier = RandomForestClassifier()
Random = RandomizedSearchCV(rf_classifier, param_distributions=rf_params,n_iter=n_iter_search,cv=cv,scoring='accuracy')
Random.fit(X, y)
time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(Random.best_params_)
print('Accuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open('results_RandomizedSearch', 'a') as results:
    results.write("\n------------------------------------------------\nRandom Forest\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(Random.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (Random.best_score_*100, mean(Random.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")

#-----------------------------------------------------------------------------------------------------------------




