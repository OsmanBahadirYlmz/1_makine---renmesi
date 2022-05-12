# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:11:28 2020

@author: oby_pc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv("musteriler.csv")

X=veriler.iloc[:,3:].values


#1.K-Means
from sklearn.cluster import KMeans

kmeans =KMeans(n_clusters=2,init="k-means++")
kmeans.fit(X)
print (kmeans.cluster_centers_)



#1.2 K-Means en uygun ayırma adeti (n_cluster,k value) bulma 
"""" n custers değerini buluyoruz wcss grafiği çiziyoruz dirsek yapan yer en
uygun cluster sasıyo dökümantasyonda ayrıntılı bilgi var
"""
sonuclar=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++", random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    
# plt.plot(range(1,10),sonuclar)
# plt.show()

#en iyi sonucu 4 olarak bulduk ve 4 ile tekrar ayıralım
kmeans=KMeans(n_clusters=4,init="k-means++", random_state=123)
Y_tahmin= kmeans.fit_predict(X)


#1.3 K-means plotlama
# plt.scatter(X[:,0],X[:,1])
# plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color=("r"))
# plt.show

plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100 ,c="red")
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100 ,c="blue")
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100 ,c="green")
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100 ,c="yellow")
plt.title("KMeans")
plt.show()






#-------------------------------
#2.hiyerarşik clustering

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=(4), affinity="euclidean",linkage="ward")

Y_tahmin=ac.fit_predict(X)#hem fit et hemde tahmin et. hangi clusterda olduğunu
print (Y_tahmin)

#alttaki iki plot aynı örnekleme olsun diye ikisinide yazdım

# plt.scatter(X[:,0],X[:,1],c=Y_tahmin)
# plt.show


plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100 ,c="blue")
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100 ,c="yellow")
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100 ,c="red")
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100 ,c="green")
plt.title("HC")
plt.show()

import scipy.cluster.hierarchy as sch

dendrogram=sch.dendrogram(sch.linkage(X,method="ward"))
plt.show()














