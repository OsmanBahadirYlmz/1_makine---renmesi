# -*- coding: utf-8 -*-
import pandas as pd


df=pd.read_csv("titanic.csv")
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values


#regresion line parametrelerini bulduk , basitleştirme için küçülttük 
# aşağıda ayrıntılısı var

from sklearn.linear_model import LogisticRegression 
model = LogisticRegression()
X = df[['Fare', 'Age']].values
y = df['Survived'].values
model.fit(X, y)

print(model.coef_, model.intercept_)
# [[ 0.01615949 -0.01549065]] [-0.51037152]

#Plotting the adjusted line over the scattered values
import matplotlib.pyplot as plt
import numpy as np
y = np.linspace(0,80,100)                     #Evenly spaced points (100) in a given interval (from 0 to 80)
x = (0.01549065*y + 0.51037152)/0.01615949    #X value taken from the line ecuation
plt.plot(x, y, '-b', label='')                #Ploting the line
plt.xlabel('Fare')
plt.ylabel('Age')
plt.scatter(df['Fare'], df['Age'], c=df['Survived'])  #plt.scatter(xAxis,yAxis,classes)
plt.grid()                                    #Grid over plotted image
plt.show()

#model predict, yukarıda daralttığımız şeyi genişlettik

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model.fit(X, y)
print ("--------")
r = X[:20]
print(model.predict([[3, True, 22.0, 1, 0, 7.25]]))
print(model.predict(X[:20]))
print(y[:20])
y_pred=model.predict(X)
c=y_pred==y
print(c.sum()) #kaçını doğru bildik
print(y.sum()) #kaç kişi hayatta
print ((y==0).sum()) #kaç kişi ölü
mask=y==0
print (mask.sum()) #kaç kişi ölü
print ((df["Survived"]==1).sum())  #kaç kişi hayatta

#Score
print (c.sum()/y.shape[0])
print("your prediction score is %" + str(model.score(X, y)*100))
