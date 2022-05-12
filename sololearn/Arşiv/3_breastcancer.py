# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 02:40:07 2020

@author: oby_pc
"""

from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
print(cancer_data.keys())
print(cancer_data['DESCR'])

import pandas as pd
df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
print(df.head())
df['target'] = cancer_data['target']


from sklearn.linear_model import LogisticRegression 
X = df[cancer_data.feature_names].values
y = df['target'].values
model = LogisticRegression(solver='liblinear')
model.fit(X, y)
print("prediction for datapoint 0:", model.predict([X[0]]))
print(model.score(X, y))