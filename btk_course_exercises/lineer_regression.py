#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 22:32:12 2024

@author: tekes
"""

# linner regresyon

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#from sklearn.datasets import 
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
x=np.random.rand(100,1)
y=5*x+5++np.random.rand(100,1)
plt.scatter(x,y)
linear_reg=LinearRegression()
linear_reg.fit(x, y)

y_pred=linear_reg.predict(x)
plt.plot(x,y_pred,color="red")

hyper_a,hyper_b=linear_reg.coef_[0],linear_reg.intercept_
print(f"{hyper_a}*x+{hyper_b}")