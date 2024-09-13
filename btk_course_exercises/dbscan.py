#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 19:49:33 2024

@author: tekes
"""
from sklearn.datasets import make_circles # kümeleme algoritmaları için dataset
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

x,_=make_circles(n_samples=1000,random_state=42,factor=0.5,noise=0.05)

dbscan=DBSCAN(eps=0.1,min_samples=5)
# eps= 2 nokta arasında koöuş kabul edilebilmesi için gerekli olan uzaklık
# min_samples= bir noktanın küme olabilmesi için gerekli olan komşu sayısı

cluster_labels=dbscan.fit_predict(x)


plt.figure()
plt.scatter(x[:,0],x[:,1],c=cluster_labels,cmap="viridis")
plt.title("dbscan results")

