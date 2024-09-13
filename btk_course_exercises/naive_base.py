#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:09:13 2024

@author: tekes
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
iris=load_iris()

x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

nb_clf=GaussianNB()

nb_clf.fit(x_train, y_train)

y_pred=nb_clf.predict(x_test)
print(classification_report(y_test, y_pred))