from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from scipy.stats import ttest_rel
from tabulate import tabulate

from scipy.stats import rankdata
from scipy.stats import ranksums


def writeToFile(stat_better_table, currentDataSet, tmpStr, scoresToFile):
    f = open("results/results_{}.txt".format(tmpStr), "a")

    f.write("\n")
    f.write(currentDataSet)
    f.write("\n")
    f.write(tmpStr)
    f.write("\n")

    for i in range(0, 9, 3):

        f.write(scoresToFile[i] + " ")
        f.write("{:.3f}".format(scoresToFile[i+1]) + " ")
        f.write("(" + "{:.2f}".format(scoresToFile[i+2]) + ")")
        f.write("\n")

    f.write("\n")
    f.write("Statistically significantly better: ")
    f.write("\n")
    f.write(stat_better_table)
    f.write("\n")

    f.close()


def writeToFile2(tmpStr, meanRanks, advantage_table, significance_table, w_statistic):
    f = open("results/results_wilcoxon_{}.txt".format(tmpStr), "a")

    f.write("\n")
    f.write("Mean ranks :")
    f.write(str(meanRanks))
    f.write("\n")
    f.write("Advantage :")
    f.write(str(advantage_table))
    f.write("\n")
    f.write("Statistical significance :")
    f.write(str(significance_table))
    f.write("\n")
    f.write("w_statistic :")
    f.write(str(w_statistic))
    f.write("\n")

    f.close()


dataSets = ['data1',
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
            'data20']

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
    estimators=base2, voting='hard')
clf3_hard = VotingClassifier(
    estimators=base3, voting='hard')

clfs1 = {
    'Z1': clf1_hard,
    'Z2': clf2_hard,
    'Z3': clf3_hard,
}

clf1_soft = VotingClassifier(
    estimators=base1, voting='soft')
clf2_soft = VotingClassifier(
    estimators=base2, voting='soft')
clf3_soft = VotingClassifier(
    estimators=base3, voting='soft')

clfs2 = {
    'Z1': clf1_soft,
    'Z2': clf2_soft,
    'Z3': clf3_soft,
}

clfs = [ clfs2]

for currentClfs in clfs:

    tmpStr = currentClfs['Z1'].voting

    for currentDataSet in dataSets:

        scoresToFile = []

        dataset = currentDataSet
        dataset = np.genfromtxt("data/%s.csv" % (dataset), delimiter=",")
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)

        n_splits = 5
        n_repeats = 2
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        scores = np.zeros((len(currentClfs), n_splits * n_repeats))

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for clf_id, clf_name in enumerate(currentClfs):
                clf = clone(currentClfs[clf_name])
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)

        mean = np.mean(scores, axis=1)
        std = np.std(scores, axis=1)

        for clf_id, clf_name in enumerate(currentClfs):
            print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))
            scoresToFile.append(clf_name)
            scoresToFile.append(mean[clf_id])
            scoresToFile.append(std[clf_id])

        np.save('results', scores)

        scores = np.load('results.npy')
        print("Folds:\n", scores)

        alfa = .05
        t_statistic = np.zeros((len(currentClfs), len(currentClfs)))
        p_value = np.zeros((len(currentClfs), len(currentClfs)))

        for i in range(len(currentClfs)):
            for j in range(len(currentClfs)):
                t_statistic[i, j], p_value[i, j] = ttest_rel(
                    scores[i], scores[j])

        headers = ["Z1", "Z2", "Z3"]
        names_column = np.array([["Z1"], ["Z2"], ["Z3"]])

        advantage = np.zeros((len(currentClfs), len(currentClfs)))
        advantage[t_statistic > 0] = 1
        advantage_table = tabulate(np.concatenate(
            (names_column, advantage), axis=1), headers)

        significance = np.zeros((len(currentClfs), len(currentClfs)))
        significance[p_value <= alfa] = 1
        significance_table = tabulate(np.concatenate(
            (names_column, significance), axis=1), headers)

        stat_better = significance * advantage
        stat_better_table = tabulate(np.concatenate(
            (names_column, stat_better), axis=1), headers)
        print("Statistically significantly better:\n", stat_better_table)

        writeToFile(stat_better_table, currentDataSet, tmpStr, scoresToFile)

    n_datasets = len(dataSets)
    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    scores = np.zeros((len(currentClfs), n_datasets, n_splits * n_repeats))

    for data_id, dataset in enumerate(dataSets):
        dataset = np.genfromtxt("data/%s.csv" %
                                (dataset), delimiter=",")
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)
        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for clf_id, clf_name in enumerate(currentClfs):
                clf = clone(currentClfs[clf_name])
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                scores[clf_id, data_id, fold_id] = accuracy_score(
                    y[test], y_pred)

    np.save('results', scores)

    scores = np.load('results.npy')
    print("\nScores:\n", scores.shape)

    mean_scores = np.mean(scores, axis=2).T
    print("\nMean scores:\n", mean_scores)

    ranks = []
    for ms in mean_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    print("\nRanks:\n", ranks)

    mean_ranks = np.mean(ranks, axis=0)
    print("\nMean ranks:\n", mean_ranks)

    alfa = .05
    w_statistic = np.zeros((len(currentClfs), len(currentClfs)))
    p_value = np.zeros((len(currentClfs), len(currentClfs)))

    for i in range(len(currentClfs)):
        for j in range(len(currentClfs)):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

    headers = list(currentClfs.keys())
    names_column = np.expand_dims(np.array(list(currentClfs.keys())), axis=1)

    advantage = np.zeros((len(currentClfs), len(currentClfs)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("\nAdvantage:\n", advantage_table)

    significance = np.zeros((len(currentClfs), len(currentClfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)

    print('Wilcoxon', w_statistic)

    writeToFile2(tmpStr, mean_ranks, advantage_table,
                 significance_table, w_statistic)
