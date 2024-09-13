#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 00:34:15 2024

@author: tekes
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier


model=load_iris()
x=model.data
y=model.target

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

knn=KNeighborsClassifier()
knn_param_grid={"n_neighbors":np.arange(2,31)}
knn_grid_search=GridSearchCV(knn,knn_param_grid)# vermiş olduğumuz sayıları tek tek dener ve en büyük olan accuracy yi döner.
# veri sayısı az olduğu durumlarda tercih edilir.
knn_grid_search.fit(x_train,y_train)

print(f"knn best parameters {knn_grid_search.best_params_}")
print(f"knn best parameters {knn_grid_search.best_score_}")

knn_random_search=RandomizedSearchCV(knn,knn_param_grid,n_iter=10)# random bi şekilde 10 tane değeri karşılaştırır ve en büyüğünü
# döner. eğer veri sayısı çok ise bu kullanılabilir.
knn_random_search.fit(x_train,y_train)

print(f"knn best parameters {knn_random_search.best_params_}")
print(f"knn best parameters {knn_random_search.best_score_}")