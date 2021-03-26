#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:54:27 2020
@author: Leila Zahedi
"""

#-------------------------------------------------------------------------------------------------------------------------
#--------------------------------Baseline Machine Learning models: Classifiers with Grid Search---------------------------
#-------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------Main Libraries-------------------------------------------------------
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix#, accuracy_score
#import scipy.stats as stats
from sklearn.model_selection import GridSearchCV
from numpy import mean
import scipy.stats as stats
from sklearn.model_selection import KFold
#----------------------------------------------------Split Dataset-------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=101)

# define cv
cv = KFold(n_splits=3, random_state=100, shuffle=True)
#create results file
open(directory, 'w').close()

#------------------------------------------------------ Naive Bayes-------------------------------------------
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
    'alpha': [round(i,1) for i in list(np.arange(0.1, 1.0, 0.1))],
    'fit_prior': [True, False],
    #'class_prior':[]
    }
    
time1 = datetime.datetime.now()# start time

mNB_classifier = MultinomialNB()
grid = GridSearchCV(mNB_classifier, rf_params, cv=cv, scoring='accuracy',verbose=4)
grid.fit(Xnb, ynb)

time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(grid.best_params_)
print('Accuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open(directory, 'a') as results:
    results.write("\n------------------------------------------------\nNaive Bayes\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(grid.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")

#--------------------------------------------------------LR-------------------------------------------
print("Logistic Regression_GridSearch")
print("------------------------------------------------")
from sklearn.linear_model import LogisticRegression

# Define the hyperparameter configuration space
rf_params = {
    'penalty' : [ 'l1', 'l2' , 'elasticnet'],
    'C' : [round(i,1) for i in list(np.arange(0, 1.0, 0.05))],
    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'random_state': [100]
    #'class_weight': ['balanced', None ]
} #150

time1 = datetime.datetime.now()# start time

lr_classifier= LogisticRegression(max_iter=10000)
grid = GridSearchCV(lr_classifier, rf_params, cv=cv, scoring='accuracy',verbose=4)
grid.fit(X, y)

time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(grid.best_params_)
print('Accuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open(directory, 'a') as results:
    results.write("\n------------------------------------------------\nLogistic Regression\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(grid.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")
    
#-------------------------------------------------------Decision Tree---------------------------------------------
print("------------------------------------------------")
print("Decision Tree")
print("------------------------------------------------")
from sklearn.tree import DecisionTreeClassifier

rf_params = {
    'max_features': ['sqrt','auto','log2',None],
    'max_depth': list(range(5, 51)),
    'min_samples_leaf': list(range(1, 16)),
    'min_samples_split': list(range(2, 31)),
    'criterion':['gini','entropy'],
    'random_state': [100]
    }

time1 = datetime.datetime.now()# start time

dtree_classifier= DecisionTreeClassifier()
grid = GridSearchCV(dtree_classifier, rf_params, cv=cv, scoring='accuracy',verbose=4)
grid.fit(X, y)

time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(grid.best_params_)
print('Accuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open(directory, 'a') as results:
    results.write("\n--------------------------------------------*GridSearch*-----------------------------------------------\n")
    results.write("\n------------------------------------------------\nDecision Tree\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(grid.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")


#--------------------------------------------------------Support Vector Machine-------------------------------------------
print("Support Vector Machine")
print("------------------------------------------------")
from sklearn.svm import SVC #,SVR

rf_params = {
    'C': [round(i,1) for i in list(np.arange(0.1, 50.0, 0.5))],
    'kernel':['linear','poly','rbf','sigmoid'],
    'random_state': [100]
}

time1 = datetime.datetime.now()# start time

svm_classifier = SVC()
grid = GridSearchCV(svm_classifier, rf_params, cv=cv, scoring='accuracy',verbose=10)
grid.fit(X, y)

time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(grid.best_params_)
print('Accuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open(directory, 'a') as results:
    results.write("\n------------------------------------------------\nSupport Vector Machine\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(grid.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")


#--------------------------------------------------------K-Nearest Neighbor-------------------------------------------
print("KNN")
print("------------------------------------------------")
from sklearn.neighbors import KNeighborsClassifier #, KNeighborsRegressor

rf_params = {
    'n_neighbors': list(range(1,21,1)),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
} #160

time1 = datetime.datetime.now()# start time

kn_classifier= KNeighborsClassifier()
grid = GridSearchCV(kn_classifier, rf_params, cv=cv, scoring='accuracy',verbose=10)
grid.fit(X, y)

time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(grid.best_params_)
print('Accuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open(directory, 'a') as results:
    results.write("\n------------------------------------------------\nK-Nearest Neighbor\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(grid.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")

#--------------------------------------------------------XGBoost-------------------------------------------
print("XGBoost")
print("------------------------------------------------")
from xgboost import XGBClassifier

y1= y.astype('category')
y1 = y1.cat.codes
y1 = y1.replace({0:1, 1:0})

rf_params = {
    'n_estimators': list(range(5,501,50)), #10
    'learning_rate': list(np.arange(0, 1.1, 0.4)),#3
    'max_depth': list(range(5,51,5)), # 10
    'subsample' : list(np.arange(0.1, 1.1, 0.4)), #3
    'colsample_bytree':list(np.arange(0.1, 1.1, 0.4)),#3
}

#X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size=0.2,random_state=100)

time1 = datetime.datetime.now()# start time

xgb_classifier = XGBClassifier(objective='binary:logistic', use_label_encoder=False, random_state=100)
grid = GridSearchCV(xgb_classifier, rf_params, cv=cv, scoring='accuracy',verbose=10)
grid.fit(X, y1)

time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(grid.best_params_)
print('Accuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open(directory, 'a') as results:
    results.write("\n------------------------------------------------\nXGBoost\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(grid.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")


#--------------------------------------------------------Random Forest-------------------------------------------
print("Random Forest_GridSearch")
print("------------------------------------------------")
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier #,RandomForestRegressor

# Define the hyperparameter configuration space

rf_params = {
    'n_estimators': list(range(5,551,50)), #10
    'max_features': ['sqrt','auto','log2'],#3
    'max_depth': list(range(5,51,5)), # 10
    'min_samples_leaf': list(range(1,16,5)),#3
    'min_samples_split':list(range(2,31,5)),#6
    'criterion':['gini','entropy'],#2
    'random_state': [100]
} 



time1 = datetime.datetime.now()# start time

rf_classifier = RandomForestClassifier()
grid = GridSearchCV(rf_classifier, rf_params, cv=cv, scoring='accuracy',verbose=10)
grid.fit(X, y)

time2 = datetime.datetime.now()# end time
duration=str(time2-time1) 

print(grid.best_params_)
print('Accuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
print("Duration:" + " " + duration)
print("------------------------------------------------")

with open(directory, 'a') as results:
    results.write("\n------------------------------------------------\nRandom Forest\n------------------------------------------------\n")
    results.write("Best Parameteres: "+ str(grid.best_params_))
    results.write("\nDuration:" + " " + duration)
    results.write('\nAccuracy: %.5f (%.5f)' % (grid.best_score_*100, mean(grid.cv_results_['std_test_score'])*100))
    results.write("\n------------------------------------------------\n")

#-----------------------------------------------------------------------------------------------------------------


