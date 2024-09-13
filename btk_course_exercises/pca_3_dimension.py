#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:26:28 2024

@author: tekes
"""

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
iris=load_iris()

x=iris.data
y=iris.target

pca=PCA(n_components=3)
x_pca=pca.fit_transform(x)

fig=plt.figure(1,figsize=(8,6))
ax=fig.add_subplot(111,projection="3d",azim=110,elev=150)

ax.scatter(x_pca[:,0],x_pca[:,1],x_pca[:,2],c=y,s=40)