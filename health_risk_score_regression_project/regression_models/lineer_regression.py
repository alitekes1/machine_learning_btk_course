#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 18:06:38 2024

@author: tekes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("data/final_df.csv")

data_columns=data.columns

lineer_reg_r2_score={}
lineer_reg_mse_score={}


for i in range(len(data_columns)):

    current_column=data_columns[i]
    x=data[current_column].values
    y=data["Health_Risk_Score"].values
    
    x=x.reshape(-1,1)
    
    x_train,x_test,y_train,y_test=train_test_split(x, y,random_state=42,test_size=0.3)
    
    
    scaler=StandardScaler()
    x_train = scaler.fit_transform(x_train)# bu sayede verilerin ortalaması 0 , varyansı 1 olur.
    x_test = scaler.transform(x_test)
    
    model=LinearRegression()
    model.fit(x_train, y_train)
    
    y_pred=model.predict(x_test)
    
    # MSE
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)  # R² Skoru
    lineer_reg_r2_score[current_column]=r2
    lineer_reg_mse_score[current_column]=mse
    
# Bar chart oluştur
plt.figure()
plt.bar(lineer_reg_r2_score.keys(),lineer_reg_r2_score.values())
plt.figure()
plt.bar(lineer_reg_mse_score.keys(),lineer_reg_mse_score.values())

