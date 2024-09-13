#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:58:58 2024

@author: tekes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

plt.figure(figsize=(6,6))
plt.subplot(1,2,1)
plt.title("Raw Data")
x,_=make_blobs(n_samples=300, centers=4,cluster_std=0.6,random_state=42)
plt.scatter(x[:,0],x[:,1])

kmeans=KMeans(n_clusters=4)
kmeans.fit(x)

plt.subplot(1,2,2)
plt.scatter(x[:,0],x[:,1],c=kmeans.labels_,cmap="viridis")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],c="red",marker="X")
plt.title("Clustering Data")