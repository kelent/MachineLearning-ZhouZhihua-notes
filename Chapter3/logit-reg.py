# -*- coding: utf-8 -*-
"""
logistic regression for watermelon dataset
Created on Fri Sep 22 14:57:43 2017

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics

dataset=np.loadtxt('3.0a.csv', delimiter=',')

X= dataset[:,1:3]
Y=dataset[:,3]

#plot dataset
f1=plt.figure(1)
plt.title('3.0a')
plt.xlabel('density')
plt.ylabel('suger_ratio')
plt.scatter(X[Y==0,0], X[Y==0,1],marker='o',color='k',s=100,label='bad')
plt.scatter(X[Y==1,0], X[Y==1,1],marker='o',color='g',s=100,label='good')
plt.legend(loc='upper right')
plt.show()

#self-coding

#likelihood
def likelihood(X,Y,beta):
      sum=0
      m,n=np.shape(X)
      
      for i in range(m):
            sum+= -Y[i]*np.dot(beta,X[i].T)+np.math.log(1+np.math.exp(np.dot(beta,X[i].T)))

      return sum
      
def sigmoid(x, beta):
      return 1.0/(1+np.math.exp(-np.dot(beta,x.T)))
      
def gradDscent(X,Y):
      
      h=0.1
      mtime=500
      m,n=np.shape(X)
      b = np.zeros((n, mtime))  #  for show convergence curve of parameter     
      
      beta=np.zeros(n)
      delta_beta=np.ones(n)*h
      llh=0
      llh_temp=0

      for i in range(mtime):
            beta_temp=beta
            
            for j in range(n):
                  beta[j]+=delta_beta[j]
                  llh_temp=likelihood(X,Y,beta)
                  delta_beta[j]=-h*(llh_temp-llh)/delta_beta[j]
                  beta[j]=beta_temp[j]

            beta+=delta_beta
            llh=likelihood(X,Y,beta)
            if delta_beta<0.0001: break
      #return beta
      
      
      #drawing
      t = np.arange(mtime)
    
      f2 = plt.figure(3) 
    
      p1 = plt.subplot(311)
      p1.plot(t, b[0])  
      plt.ylabel('w1')  
    
      p2 = plt.subplot(312)
      p2.plot(t, b[1])  
      plt.ylabel('w2')  
        
      p3 = plt.subplot(313)
      p3.plot(t, b[2])  
      plt.ylabel('b')  
        
      plt.show()               
      return beta
      
def predict(X,beta):
     m,n=np.shape(X)
     y=np.zeros(m)
     
     for i in range(m):
           if sigmoid(X[i],beta)>0.5: y[i]=1;
     return y
     
#sklearn



#split train set and test set
X_train, X_test, Y_train, Y_test=model_selection.train_test_split(X,Y,test_size=0.5,random_state=0)

log_model=linear_model.LogisticRegression()
log_model.fit(X_train,Y_train)

Y_pred=log_model.predict(X_test)

print(metrics.confusion_matrix(Y_test,Y_pred))
print(metrics.classification_report(Y_test,Y_pred))

