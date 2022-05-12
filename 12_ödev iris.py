# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:16:03 2020

@author: sadievrenseker
"""


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_excel('Iris.xls')
#pd.read_csv("veriler.csv")
#test


x = veriler.iloc[:,0:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken




#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, np.ravel(y_train))

y_pred = logr.predict(X_test)
# print(y_pred)
# print(y_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("logistic regresion \n" , cm)


print("accuracy:", accuracy_score(y_test, y_pred))
# print("precision:", precision_score(y_test, y_pred,average='micro'))
# print("recall:", recall_score(y_test, y_pred))
# print("f1 score:", f1_score(y_test, y_pred))


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(X_train, np.ravel(y_train))

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(" KNeighborsClassifier \n",  cm)



from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, np.ravel(y_train))

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)


print('SVC')
print(cm)

df = pd.DataFrame(x_test, columns=["0","1","2","3"])
df["actual"] = y_test
df["predicted"] = y_pred

incorrect = df[df["actual"] != df["predicted"]]

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train,  np.ravel(y_train))

# y_pred = gnb.predict(X_test)

# cm = confusion_matrix(y_test,y_pred)
# print('GNB')
# print(cm)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train, np.ravel(y_train))
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train, np.ravel(y_train))

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)


    
# # 7. ROC , TPR, FPR değerleri 

# y_proba = rfc.predict_proba(X_test)
# # print(y_test)
# print(y_proba[:,0])

# from sklearn import metrics
# fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
# print(fpr)
# print(tpr)









