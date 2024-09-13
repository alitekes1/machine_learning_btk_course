#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 20:22:44 2024

@author: tekes
"""

#polinomiyal regression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x=4*np.random.rand(100,1)
y=2+3*x**2

poly_fit=PolynomialFeatures(degree=2)# degree hyperparametredir.
x_tansform=poly_fit.fit_transform(x)

pol_reg=LinearRegression()
pol_reg.fit(x_tansform,y)

x_test=np.linspace(0,4, 100).reshape(-1,1)
x_test_poly=poly_fit.transform(x_test)

y_pred=pol_reg.predict(x_test_poly)
plt.plot(x_test,y_pred)
plt.scatter(x,y)
