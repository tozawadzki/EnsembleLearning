from sklearn.ensemble import VotingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataSetsNames = [
    'data1',
    'data2',
    'data3',
    'data4',
    'data5',
    'data6',
    'data7',
    'data8',
    'data9',
    'data10',
    'data11',
    'data12',
    'data13',
    'data14',
    'data15',
    'data16',
    'data17',
    'data18',
    'data19',
    'data20',
]


def writeToFile(soft_accuracy, hard_accuracy):
    f = open("results.txt", "a")
    f.write(x)
    f.write("\n")
    f.write("Soft voting: ")
    f.write(str(soft_accuracy))
    f.write("\n")
    f.write("Hard voting: ")
    f.write(str(hard_accuracy))
    f.write("\n")
    f.close()


def createPlot(soft_predict, Y_test):
    plt.figure(figsize=(5, 5))
    plt.scatter(Y_test, soft_predict, c='crimson')

    p1 = max(max(soft_predict), max(Y_test))
    p2 = min(min(soft_predict), min(Y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=10)
    plt.ylabel('Predictions', fontsize=10)
    plt.axis('equal')
    plt.show()


def ensemble_voting(x):
    base_classifiers = []

    base_classifiers.append(('KNN', KNeighborsClassifier()))
    base_classifiers.append(('DTC', DecisionTreeClassifier()))
    base_classifiers.append(('LR', LogisticRegression(max_iter=1000000)))
    base_classifiers.append(('RF', GaussianNB()))
    base_classifiers.append(('SVC', SVC(gamma="auto", probability=True)))

    ensemble_voting_soft = VotingClassifier(
        estimators=base_classifiers, voting='soft')
    ensemble_voting_soft.fit(X_train, Y_train)

    ensemble_voting_hard = VotingClassifier(
        estimators=base_classifiers, voting='hard')
    ensemble_voting_hard.fit(X_train, Y_train)

    soft_predict = ensemble_voting_soft.predict(X_test)
    hard_predict = ensemble_voting_hard.predict(X_test)

    soft_accuracy = accuracy_score(Y_test, soft_predict)
    hard_accuracy = accuracy_score(Y_test, hard_predict)

    print("Prediction")
    print("Data set:   ", Y_test)
    print("Soft voting:", soft_predict, "\nHard voting:", hard_predict)
    print("Accuracy")
    print("Soft voting:", soft_accuracy, "\nHard voting:", hard_accuracy)

    writeToFile(soft_accuracy, hard_accuracy)
    #createPlot(soft_predict, Y_test)


for x in dataSetsNames:

    df = pd.read_excel("data\{}.xls".format(x), sheet_name="Sheet1")
    X = np.array(df, dtype='float32')
    Y = np.array(pd.read_excel("data\{}.xls".format(x), sheet_name="Sheet2"))
    Y = Y.ravel()

    print(Y.size)

    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        Y,
                                                        test_size=0.20,
                                                        random_state=42)

    ensemble_voting(x)
