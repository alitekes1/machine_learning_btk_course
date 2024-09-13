#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:47:47 2024

@author: tekes
"""

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

diabetes=load_diabetes()

X=diabetes.data
y=diabetes.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

tree_reg=DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
y_predict=tree_reg.predict(X_test)
mean_error=mean_squared_error(y_test, y_predict)
print(f"root mean squred error: {np.sqrt(mean_error)}")
