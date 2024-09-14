#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 00:18:38 2024

@author: tekes
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.svm import SVC

data_train=pd.read_csv("data/train_preprocessed.csv")
data_test=pd.read_csv("data/test_preprocessed.csv")

x_train=data_train.drop("satisfaction",axis=1).values
x_test=data_test.drop("satisfaction",axis=1).values

y_train=data_train["satisfaction"].values
y_test=data_test["satisfaction"].values

model=SVC()

model.fit(x_train[0:10390], y_train[0:10390])

y_pred=model.predict(x_test)

score=model.score(x_test[0:2597], y_test[0:2597])
report=classification_report(y_test, y_pred)
print(score)
print(report)