#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:32:28 2024

@author: tekes
"""

from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
mnist=fetch_openml("mnist_784", version=1)

x=mnist.data
y=mnist.target.astype(int)

lda=LinearDiscriminantAnalysis(n_components=2)# 2 boyuta indirgeyeceğiz.
x_lda= lda.fit_transform(x,y)

plt.figure()
plt.scatter(x_lda[:,0],x_lda[:,1],c=y,cmap="tab10",alpha=0.6)
plt.colorbar(label="Digits")