from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import sklearn.linear_model as lm


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')

def result(request):
    dataset = pd.read_csv('diabetes.csv')
    X = dataset.drop(columns='Outcome', axis=1)
    Y = dataset['Outcome']
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    x = standardized_data
    y = dataset['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(x_train, y_train)
    x_train_prediction = classifier.predict(x_train)
    training_data_accuracy = accuracy_score(x_train_prediction, y_train)

    value1 = float(request.GET['n1'])
    value2 = float(request.GET['n2'])
    value3 = float(request.GET['n3'])
    value4 = float(request.GET['n4'])
    value5 = float(request.GET['n5'])
    value6 = float(request.GET['n6'])
    value7 = float(request.GET['n7'])
    value8 = float(request.GET['n8'])

    DPS_predict = classifier.predict([[value1, value2, value3, value4, value5, value6, value7, value8]])

    result1 = ""
    if DPS_predict == [1]:
        result1 = "Diabetics positive"
    else:
        result1 = "Diabetics negative"

    return render(request, 'predict.html', {'result2': result1})
