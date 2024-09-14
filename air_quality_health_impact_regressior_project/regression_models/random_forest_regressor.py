#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 18:55:55 2024

@author: tekes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("data/final_df.csv")

x=data.drop("Health_Risk_Score",axis=1).values
y=data.Health_Risk_Score.values

x_train,x_test,y_train,y_test=train_test_split(x,y, random_state=42,test_size=0.3)

scaler=StandardScaler()

x_train = scaler.fit_transform(x_train)# bu sayede verilerin ortalaması 0 , varyansı 1 olur.
x_test = scaler.transform(x_test)

model=RandomForestRegressor(n_estimators=10,max_depth=10)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R² Skoru
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² Score: {r2}")

def random_forest_reg():
    return r2,rmse,mse