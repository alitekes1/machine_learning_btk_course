#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 22:47:17 2024

@author: tekes
"""
# linner regresyon2

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
#from sklearn.datasets import 
from sklearn.metrics import accuracy_score,mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

diabet=load_diabetes()

x=diabet.data[:,np.newaxis,2]
y=diabet.target

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.1)

model=LinearRegression()
model.fit(x_train, y_train)

pred_func=model.predict(x_test)

plt.figure()
plt.plot(x_test, pred_func,color="red")
plt.scatter(x, y)

mse=mean_squared_error(y_test, pred_func)
hyper_a,hyper_b=model.coef_[0],model.intercept_
print(f"{hyper_a}*x+{hyper_b}")
print(f"mse:{mse}")