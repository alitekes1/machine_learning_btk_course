#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:15:16 2024

@author: tekes
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.datasets import fetch_olivetti_faces #dataset  
from sklearn.ensemble import RandomForestClassifier# random forest i uygulamak için gereklidir.

oli=fetch_olivetti_faces()

x=oli.data
y=oli.target

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

classifier=RandomForestClassifier(n_estimators=100,random_state=42)
#içerisinde bulunan decision treenin adedi ve bu tree lerin random olma derecesi hyperparametredir.
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

mse=mean_squared_error(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)
print(mse)
print(f"accuracy percent:{accuracy*100}")