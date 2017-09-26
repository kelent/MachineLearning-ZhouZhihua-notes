# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:35:17 2017

@author: Administrator
"""

#Iris data cross-validation

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn import datasets
import numpy as np
import pandas as pd

#tmp=pd.read_csv('iris.data',sep=',')
#iris=np.loadtxt('iris.data', delimiter=',')
iris=datasets.load_iris()
x=iris['data'][0:149]
y=iris['target'][0:149]


log_model=LogisticRegression()
m=np.shape(x)[0]

y_pred=cross_val_predict(log_model,x,y,cv=10)
print(metrics.accuracy_score(y,y_pred))
#print(y_pred)

loo=LeaveOneOut()
accuracy=0
for train,test in loo.split(x):
      log_model.fit(x[train],y[train])
      y_pred1=log_model.predict(x[test])
      if y_pred1==y[test]:accuracy+=1
print (accuracy/m)