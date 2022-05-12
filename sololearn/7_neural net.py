# -*- coding: utf-8 -*-

#1.Creating Artificial dataset
"""
• n_samples: number of datapoints
• n_features: number of features
• n_informative: number of informative features
• n_redundant: number of redundant features
• random_state: random state to guarantee same result every time

"""
#rastgele bir dataset oluşturduk. yukarıda özellikleri var
from sklearn.datasets import make_classification
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=3)
print(X)
print(y)

from matplotlib import pyplot as plt
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], s=100, edgecolors='y')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], s=100, edgecolors='k', marker='^')
plt.show()

#1.1 MLP classifier

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
mlp = MLPClassifier(max_iter=1000) # The default number of iterations is 200. Let’s up this value to 1000.
mlp.fit(X_train, y_train)
print("accuracy:", mlp.score(X_test, y_test))

#1.2 İmprove MLP
"""
  The default value of alpha is 0.0001. Note that decreasing alpha often requires an increase in max_iter.
  The options for solver are 'lbfgs', 'sgd' and 'adam'.
  random_state to ensure that every time you run the code with the same parameters you will get the same output.
"""
mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50), alpha=0.0001, solver='adam', random_state=3)
mlp.fit(X_train, y_train)
print("accuracy:", mlp.score(X_test, y_test))

#2. Predicting Handwritten Digits

#2.1 import data set n_class=2 o yüzden sadece 0 ve 1 var
from sklearn.datasets import load_digits
X, y = load_digits(n_class=2, return_X_y=True)
print(X.shape, y.shape)
print(X[0])
print(y[0])
print(X[0].reshape(8, 8)) #daha iyi anlamak için8x8 yaptık

#2.2 Drawing the Digits-matshow
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

X, y = load_digits(n_class=2, return_X_y=True)
plt.matshow(X[0].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())  # remove x tick marks
plt.yticks(())  # remove y tick marks
plt.show()

#2.3 using MLP on Dataset

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
mlp = MLPClassifier(max_iter=200) # The default number of iterations is 200. Let’s up this value to 1000.
mlp.fit(X_train, y_train)
print("accuracy:", mlp.score(X_test, y_test))

from sklearn.metrics import confusion_matrix
y_pred=mlp.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("neural network \n" , cm)


#2.4 sadece 0 ve 1 i denemiştik şimdi 10 digits

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
mlp = MLPClassifier(random_state=2)
mlp.fit(X_train, y_train)

print("accuracy:", mlp.score(X_test, y_test))
y_pred=mlp.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("neural network \n" , cm)

#2.5 hata yaptığımız satırlar

y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]

j = 0
for j in range(len(incorrect)):
    plt.matshow(incorrect[j].reshape(8, 8), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())
    plt.title("true{}, pred{}".format(incorrect_true[j],incorrect_pred[j]))
    plt.show()
    print(j)
    print("true value:", incorrect_true[j])
    print("predicted value:", incorrect_pred[j])


#3 Visualizing MLP Weights

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

#3.1 kütüphane
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

print(X.shape, y.shape)
print(np.min(X), np.max(X))
print(y[0:5])

#3.2 3 ten küçükleri aldık
X5 = X[y <= '3']
y5 = y[y <= '3']

mlp=MLPClassifier(
  hidden_layer_sizes=(6,), 
  max_iter=200, alpha=1e-4,
  solver='sgd', random_state=2)

mlp.fit(X5, y5)

#3.3 coefler datalar 0-hidden layer coef, 1-outputlayer coef dökümanda örnek var

print(mlp.coefs_)
print(len(mlp.coefs_))
print(mlp.coefs_[0].shape)
coefs =mlp.coefs_

#node ları görsel hale getirerek
from matplotlib import pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(5, 4))
for i, ax in enumerate(axes.ravel()):
    coef = mlp.coefs_[0][:, i]
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(i + 1)
plt.show()















