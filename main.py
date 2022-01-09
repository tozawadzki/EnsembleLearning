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
#Sztuczna sieć neuronowa
from sklearn.neural_network import MLPClassifier
#Las losowy
from sklearn.ensemble import RandomForestClassifier
#Bayes
from sklearn.naive_bayes import GaussianNB
#Bagging, boosting?

#Dane
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#loading data
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size = 0.20, 
                                                    random_state = 42)

#sample voting from geeksforgeeks
def voting_sample():
    estimator = []
    #estimator.append(('MLP', MLPClassifier()))
    #estimator.append(('KNN', KNeighborsClassifier()))
    estimator.append(('DTC', DecisionTreeClassifier()))
    estimator.append(('SVC', SVC(gamma ='auto', probability = True)))

    vot_hard = VotingClassifier(estimators = estimator, voting ='hard')
    vot_hard.fit(X_train, y_train)
    y_pred = vot_hard.predict(X_test)

    score = accuracy_score(y_test, y_pred)
    print("Hard Voting Score % d" % score)

    vot_soft = VotingClassifier(estimators = estimator, voting ='soft')
    vot_soft.fit(X_train, y_train)
    y_pred = vot_soft.predict(X_test)

    score = accuracy_score(y_test, y_pred)
    print("Soft Voting Score % d" % score)

#knn voting for 1, 3, 5, 7, 9
#returns hard_knn, soft_nn
def knn_voting():

    def append_knn():
        models = list()
        models.append(('knn1', KNeighborsClassifier(n_neighbors=1)))
        models.append(('knn3', KNeighborsClassifier(n_neighbors=3)))
        models.append(('knn5', KNeighborsClassifier(n_neighbors=5)))
        models.append(('knn7', KNeighborsClassifier(n_neighbors=7)))
        models.append(('knn9', KNeighborsClassifier(n_neighbors=9)))
        return models

    def get_hard_knn():
        models = append_knn()
        return VotingClassifier(estimators=models, voting='hard')

    def get_soft_knn():
        models = append_knn()
        return VotingClassifier(estimators=models, voting='soft')

    knn_hard = get_hard_knn()
    knn_soft = get_soft_knn()

    knn_hard.fit(X_train, y_train)
    knn_soft.fit(X_train, y_train)

    knn_hard_pred = knn_hard.predict(X_test)
    knn_soft_pred = knn_soft.predict(X_test)
    print("Hard voting prediction: ", knn_hard_pred, "\nSoft voting prediction:", knn_soft_pred)

def ensemble_voting():
    estimator = []
    estimator.append(('DTC', DecisionTreeClassifier().fit(X_train, y_train)))
    estimator.append(('KNN', KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)))
    estimator.append(('SVC', SVC(kernel='rbf', probability=True).fit(X_train, y_train)))
    #What about these weights? VotingClassifier(estimators, voting='soft', weights=[2, 1, 2])

    ensemble_voting_soft = VotingClassifier(estimators = estimator, voting='soft')
    ensemble_voting_soft.fit(X_train, y_train)

    ensemble_voting_hard = VotingClassifier(estimators = estimator, voting='hard')
    ensemble_voting_hard.fit(X_train, y_train)

    soft_predict = ensemble_voting_soft.predict(X_test)
    hard_predict = ensemble_voting_hard.predict(X_test)

    print("Soft voting prediction: ", soft_predict, "\nHard voting prediction:", hard_predict)

#voting_sample()
#knn_voting()
ensemble_voting()