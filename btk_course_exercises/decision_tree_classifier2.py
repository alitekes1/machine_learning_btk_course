#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 00:29:14 2024

@author: tekes
"""

from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# custom data set
X=np.sort(5*np.random.rand(80,1),axis=0)
y=np.sin(X).ravel()

y[::5]+=0.5*(0.5-np.random.rand(16))# outlier ekledik
#plt.scatter(X,y)# noktalı bir şekilde grafik oluşturduk.

reg1=DecisionTreeRegressor(max_depth=5)
reg2=DecisionTreeRegressor(max_depth=15)
reg1.fit(X,y)
reg2.fit(X,y)

X_test=np.arange(0,5,0.05)[:,np.newaxis]

pred1=reg1.predict(X_test)
pred2=reg2.predict(X_test)

plt.figure()
plt.plot(X_test,pred2,label=f"pred2 depth:{reg2.max_depth}")
plt.plot(X_test,pred1,label=f"pred1 depth:{reg1.max_depth}")
plt.plot(X, y,label="original data")
plt.legend()