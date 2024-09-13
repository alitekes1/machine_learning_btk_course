#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:49:09 2024

@author: tekes
"""
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,root_mean_squared_error,accuracy_score

heart_diseasa=fetch_ucirepo(name="heart_disease")

dataframe=pd.DataFrame(data=heart_diseasa.data.features)

merged_frame=dataframe.join(heart_diseasa.data.targets)

merged_frame.dropna(inplace=True)

dataframe=merged_frame

x=dataframe.drop(["num"],axis=1).values
y=dataframe["num"].values

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.1)

log_reg=LogisticRegression(penalty="l2",C=1,solver="lbfgs",max_iter=100)

log_reg.fit(x_train, y_train)
y_pred=log_reg.predict(x_test)
accu=accuracy_score(y_test,y_pred)
print(f"accurancy value={accu}")