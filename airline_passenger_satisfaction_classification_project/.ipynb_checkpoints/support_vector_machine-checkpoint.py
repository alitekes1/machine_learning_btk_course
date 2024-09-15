#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 00:18:38 2024

@author: tekes
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

x_train=pd.read_csv("data/train.csv")
x_test=pd.read_csv("data/test.csv")

x_train.drop(columns=x_train.columns[0],inplace=True,axis=1)
x_test.drop(columns=x_test.columns[0],inplace=True,axis=1)