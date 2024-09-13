#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:31:57 2024

@author: tekes
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("data/final_df.csv")
x=df.drop("Health_Risk_Score",axis=1).values
y=df["Health_Risk_Score"].values 

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.3)

scaler=StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model=DecisionTreeRegressor(max_depth=10,random_state=42)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)


# MSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R² Skoru

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² Score: {r2}")
