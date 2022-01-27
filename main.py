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
from scipy.stats import ttest_rel
from tabulate import tabulate

datasets = [
    'data1',
]

base1 = []
base2 = []
base3 = []

base1.append(('KNN', KNeighborsClassifier()))
base1.append(('DTC', DecisionTreeClassifier()))
base1.append(('LR', LogisticRegression(max_iter=1000000)))

base2.append(('LR', LogisticRegression(max_iter=1000000)))
base2.append(('GNB', GaussianNB()))
base2.append(('SVC', SVC(gamma="auto", probability=True)))

base3.append(('KNN', KNeighborsClassifier()))
base3.append(('DTC', DecisionTreeClassifier()))
base3.append(('LR', LogisticRegression(max_iter=1000000)))
base3.append(('GNB', GaussianNB()))
base3.append(('SVC', SVC(gamma="auto", probability=True)))

clf1_hard = VotingClassifier(
    estimators=base1, voting='hard')
clf2_hard = VotingClassifier(
    estimators=base1, voting='hard')
clf3_hard = VotingClassifier(
    estimators=base1, voting='hard')

clfs1 = {
    'Z1': clf1_hard,
    'Z2': clf2_hard,
    'Z3': clf3_hard,
}

clf1_soft = VotingClassifier(
    estimators=base1, voting='soft')
clf2_soft = VotingClassifier(
    estimators=base1, voting='soft')
clf3_soft = VotingClassifier(
    estimators=base1, voting='soft')

clfs2 = {
    'Z1': clf1_soft,
    'Z2': clf2_soft,
    'Z3': clf3_soft,
}

n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)

scores = np.zeros((len(clfs1), n_datasets, n_splits * n_repeats))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("data/%s.csv" %
                            (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs1):
            clf = clone(clfs1[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)

scores = np.load('results.npy')
print("\nScores:\n", scores.shape)

mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)


alfa = .05
t_statistic = np.zeros((len(clfs1), len(clfs1)))
p_value = np.zeros((len(clfs1), len(clfs1)))

for i in range(len(clfs1)):
    for j in range(len(clfs1)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])

headers = ["Z1", "Z2", "Z3"]
names_column = np.array([["Z1"], ["Z2"], ["Z3"]])

advantage = np.zeros((len(clfs1), len(clfs1)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)

significance = np.zeros((len(clfs1), len(clfs1)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table)
