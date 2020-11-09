# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 00:38:43 2020

@author: Rewat
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv(r'C:\Users\Rewat\Desktop\data science\Iris.csv')
X=df.iloc[:,1:5].values

from sklearn.cluster import KMeans
wcs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcs.append(kmeans.inertia_)
plt.plot(range(1, 11), wcs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()