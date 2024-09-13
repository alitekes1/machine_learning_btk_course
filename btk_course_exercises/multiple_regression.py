#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 23:00:50 2024

@author: tekes
"""
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

x=np.random.rand(100,2)
coef=np.array([3,5])
y=np.dot(x,coef)

fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.scatter(x[:,0],x[:,1],y)

lin_reg=LinearRegression()
lin_reg.fit(x, y)

x1,x2=np.meshgrid(np.linspace(0, 1,10),np.linspace(0,1,10))
y_pred=lin_reg.predict(np.array([x1.flatten(),x2.flatten()]).T)
ax.plot_surface(x1,x2,y_pred.reshape(x1.shape),alpha=0.3)