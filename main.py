# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 23:00:50 2022

@author: aiatul
"""
import pandas as pd
#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("diabetes.csv")

sns.heatmap(data.isnull())

correlation = data.corr()

sns.heatmap(correlation)

x = data.drop("Outcome", axis=1)
y = data["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
accuracy = accuracy_score(prediction, y_test)

print(accuracy)
