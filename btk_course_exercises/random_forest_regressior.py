#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:32:53 2024

@author: tekes
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,root_mean_squared_error
from sklearn.datasets import fetch_california_housing #dataset  
from sklearn.ensemble import RandomForestRegressor# random forest i uygulamak için gereklidir.

california=fetch_california_housing()

x=california.data
y=california.target

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

random_forest_reg=RandomForestRegressor(random_state=42,n_estimators=100)

random_forest_reg.fit(x_train, y_train)

y_pred=random_forest_reg.predict(x_test)

mse=mean_squared_error(y_test, y_pred)
rmse=root_mean_squared_error(y_test, y_pred)
print(mse)
print(np.sqrt(rmse))