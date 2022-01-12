#Głosowanie
from sklearn.ensemble import VotingClassifier

#Klasyfikatory - różne rodzaje klasyfikatorów bazowych

#Drzewa
from sklearn.tree import DecisionTreeClassifier
#KNN
from sklearn.neighbors import KNeighborsClassifier
#Wektor wsparcia
from sklearn.svm import SVC
#Regresja 
from sklearn.linear_model import LogisticRegression
#Bayes
from sklearn.naive_bayes import GaussianNB
#Sztuczna sieć neuronowa
from sklearn.neural_network import MLPClassifier

#Do wczytywania danych z excela
import pandas as pd
import numpy as np

#Dane
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Wczytywanie danych z Excela test
df = pd.read_excel("data\data1.xls", sheet_name="Sheet1")
X = np.array(df, dtype='float32')
Y = np.array(pd.read_excel("data\data1.xls", sheet_name="Sheet2"))
Y = Y.ravel()

X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size = 0.20, 
                                                    random_state = 42)
def ensemble_voting():
    estimator = []

    estimator.append(('DTC', DecisionTreeClassifier(random_state=0)))
    estimator.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
    estimator.append(('SVC', SVC(gamma='auto', probability=True)))
    estimator.append(('LR', LogisticRegression(random_state=10, max_iter=1000)))
    estimator.append(('RF', MLPClassifier(hidden_layer_sizes=(10), max_iter=10000, random_state=0)))

    ensemble_voting_soft = VotingClassifier(estimators = estimator, voting='soft')
    ensemble_voting_soft.fit(X_train, Y_train)

    ensemble_voting_hard = VotingClassifier(estimators = estimator, voting='hard')
    ensemble_voting_hard.fit(X_train, Y_train)

    soft_predict = ensemble_voting_soft.predict(X_test)
    hard_predict = ensemble_voting_hard.predict(X_test)

    soft_accuracy = accuracy_score(Y_test, soft_predict)
    hard_accuracy = accuracy_score(Y_test, hard_predict)

    print("Prediction")
    print("Soft voting: ", soft_predict, "\nHard voting: ", hard_predict)
    print("Data set:    ", Y_test)
    print("Accuracy")
    print("Soft voting: ", soft_accuracy, "\nHard voting: ", hard_accuracy)

ensemble_voting()