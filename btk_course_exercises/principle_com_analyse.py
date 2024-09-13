#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:06:01 2024

@author: tekes
"""
# pca nın amacı boyut indirgemedir. 
# iris veri setindeki 4 boyutlu veri 2 boyuta indirgenmiştir.

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
iris=load_iris()

x=iris.data
y=iris.target

pca=PCA(n_components=2)
x_pca=pca.fit_transform(x)

plt.figure()
for i in range(len(iris.target_names)):
    plt.scatter(x_pca[y==i,0],x_pca[y==i,1],label=iris.target_names[i])