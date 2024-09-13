#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 20:52:36 2024

@author: tekes
"""

#sınıflandırma çalışması

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # classifier
from sklearn.metrics import classification_report#sınıflandırma problemleri için genellikle bu kullanılır.
model=load_digits()

x=model.data
y=model.target
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

svm_clas=SVC(random_state=42,kernel="linear")

svm_clas.fit(x_train, y_train)

y_pred=svm_clas.predict(x_test)
accu2=classification_report(y_test, y_pred)
print(accu2)