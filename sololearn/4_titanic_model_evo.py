# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression 

df=pd.read_csv("titanic.csv")
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model = LogisticRegression()
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model.fit(X, y)
y_pred=model.predict(X)

"""
#calculating metrix_score,
accuracy (.score , accuracy_score(x,y) )- percent of predictions that re correct
    Of 10000 credit card chards, we1 have 9900
    legitimate charges and 100 fraud1ulent charges.
    I could build a model that just predicts that 
    every single charge is legitimate and it would
    get 9900/10000 (99%) of the predictions correct!
accuracy=(TP+TN)/total

precision refers to the percentage of positive results which are relevant
TP/predicted yes

Recall(sensitivity) TPR- (true positive rate) is the percent of positive cases that the model predicts correctly.
=tpr=TP/(TP+FN)= TP/Actual yes

False positive rate, specisify, FPR = FP/Actual no

prevalence actual yes/total

F1 - 2*(precision*recall) / (precision+recall)
""" 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("accuracy:", accuracy_score(y, y_pred))
print("precision:", precision_score(y, y_pred))
print("recall:", recall_score(y, y_pred))
print("f1 score:", f1_score(y, y_pred))

"""
confusion matrix
A true positive (TP) is a datapoint we predicted positively that we were correct about.
A true negative (TN) is a datapoint we predicted negatively that we were correct about.
A false positive (FP) is a datapoint we predicted positively that we were incorrect about.
A false negative (FN) is a datapoint we predicted negatively that we were incorrect about.
                    (actual +) (actual -)
predicted positive     TP          FP
predicted negative     FN          TN
                    (total +)  (total -)
                    
skilearn reverse it (doğrulandı +1)

                 predicted -   predicted +
   (actual -)       TN          FP
   (actual +)       FN          TP
                    (total +)  (total -)
                    
cm matrix te her    satır toplamı gerçek değer,
                    sutun toplamı tahmin edilen değer
                                pred 0  pred 1  pred 2
                    actual 0    3       2       1
                    actual 1    0       5       2
                    actual 2    1       3       4
                    
                    
0 dan gerçekte  6 adet var 
1 den gerçekte  7 adet
2 den gerçekte  8 adet
biz 4 adet 0 tahmin etmişiz, bunların 3 ü  doğru 0(TN - 0 negatif) 1i yanlış 2
biz 10 adet 1 tahmin etmişiz bunların 5 i doğru (TP- 1 pozitif çünkü)2 tanesi 0 3 tanesi 2.toplam 5 yanlış
biz 7 adet 2 tahmin etmişiz bunların 4 ü doğru. 1 tanesi 0 ,2 tanesi 1 toplam 3 yanlış
"""

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_pred))

#train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

print("whole dataset:", X.shape, y.shape)
print("training set:", X_train.shape, y_train.shape)
print("test set:", X_test.shape, y_test.shape)

#test_train model
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
y_pred = model.predict(X_test)

print(" accuracy: {0:.5f}".format(accuracy_score(y_test, y_pred)))
print("precision: {0:.5f}".format(precision_score(y_test, y_pred)))
print("   recall: {0:.5f}".format(recall_score(y_test, y_pred)))
print(" f1 score: {0:.5f}".format(f1_score(y_test, y_pred)))

#sensivity and specifity

from sklearn.metrics import recall_score
from sklearn.metrics import recall_score, precision_recall_fscore_support
sensitivity_score = recall_score
def specificity_score(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    return r[0]

print("sensitivity:", sensitivity_score(y_test, y_pred))
print("specificity:", specificity_score(y_test, y_pred))

    #The second array is the recall, so we can ignore the other three arrays. 
    #There are two values. The first is the recall of the negative class 
    #and the second is the recall of the positive class.
    #The first value is the specificity
    #second value is the standard recall or sensitivity value
print(precision_recall_fscore_support(y_test, y_pred))


#Adjusting the Logistic Regression Threshold in Sklearn

print("predict proba:")
print(model.predict_proba(X_test))

y_pred = model.predict_proba(X_test)[:, 1] > 0.75

print("precision:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred))


#The ROC Curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.show()

print("sensitivity:", sensitivity_score(y_test, y_pred))
print("specificity:", specificity_score(y_test, y_pred))

# we calculate the Area Under the ROC Curve, also called the AUC.
# This is the area under the ROC curve. It’s a value between 0 and 1, the higher the better.
model1 = LogisticRegression()
model1.fit(X_train, y_train)
y_pred_proba1 = model1.predict_proba(X_test)
print("model 1 AUC score:", roc_auc_score(y_test, y_pred_proba1[:, 1]))

model2 = LogisticRegression()
model2.fit(X_train[:, 0:2], y_train)
y_pred_proba2 = model2.predict_proba(X_test[:, 0:2])
print("model 1 AUC score:", roc_auc_score(y_test, y_pred_proba2[:, 1]))


#K-FOLD The k is the number of chunks we split our dataset into.
# The standard number is 5
#kf normal split işlemi yapıyor ama 3 defa,çoklu tren

from sklearn.model_selection import KFold
X = df[['Age', 'Fare']].values[:6]
y = df['Survived'].values[:6]


kf = KFold(n_splits=3, shuffle=True)
chunks=kf.split(X) #chunks generator-dont need
print(list(kf.split(X)))
#-------------------------
splits = list(kf.split(X))
first_split = splits[0]
train_indices, test_indices = first_split
print("training set indices:", train_indices)
print("test set indices:", test_indices)

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]
print("X_train")
print(X_train)
print("y_train", y_train)
print("X_test")
print(X_test)
print("y_test", y_test)

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

#her bir splitti loopla test edelim
import numpy as np

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

scores = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))  
print("----------")
print(scores)
print(np.mean(scores))
final_model = LogisticRegression()
final_model.fit(X, y)

#model comparison

kf = KFold(n_splits=5, shuffle=True)

X1 = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
X2 = df[['Pclass', 'male', 'Age']].values
X3 = df[['Fare', 'Age']].values
y = df['Survived'].values

def score_model(X, y, kf):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    print("accuracy:", np.mean(accuracy_scores))
    print("precision:", np.mean(precision_scores))
    print("recall:", np.mean(recall_scores))
    print("f1 score:", np.mean(f1_scores))

print("Logistic Regression with all features")
score_model(X1, y, kf)
print()
print("Logistic Regression with Pclass, Sex & Age features")
score_model(X2, y, kf)
print()
print("Logistic Regression with Fare & Age features")
score_model(X3, y, kf)







#