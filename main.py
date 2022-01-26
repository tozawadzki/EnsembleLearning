from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone

averageResultsSoft = [[], [], []]
averageResultsHard = [[], [], []]


datasets = [
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

for x in datasets:
    def writeToFile(soft_accuracy, hard_accuracy, y):
        f = open("results/results_base_classifiers{}.txt".format(y+1), "a")
        f.write(x)
        f.write("\n")
        f.write("Soft voting: ")
        f.write(str(soft_accuracy))
        f.write("\n")
        f.write("Hard voting: ")
        f.write(str(hard_accuracy))
        f.write("\n")
        f.close()

baseClassifiers = []

baseClassifiers.append(('KNN', KNeighborsClassifier()))
baseClassifiers.append(('DTC', DecisionTreeClassifier()))
baseClassifiers.append(('LR', LogisticRegression(max_iter=1000000)))
baseClassifiers.append(('GNB', GaussianNB()))
baseClassifiers.append(('SVC', SVC(gamma="auto", probability=True)))

ensemble_voting_soft = VotingClassifier(
    estimators=baseClassifiers, voting='soft')
ensemble_voting_hard = VotingClassifier(
    estimators=baseClassifiers, voting='hard')

clfs = {
    'SOFT': ensemble_voting_soft,
    'HARD': ensemble_voting_hard,
}

n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("data/csv/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            print(clf)
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)

scores = np.load('results.npy')
print("\nScores:\n", scores.shape)

mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)
