"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy
import copy
from sklearn import naive_bayes
from sklearn import tree
from sklearn import metrics

from gspan_mining import gSpan
from gspan_mining import GraphDatabase
from bisect import insort, bisect_left


class PatternGraphs:
    """
    This template class is used to define a task for the gSpan implementation.
    You should not modify this class but extend it to define new tasks
    """
    def __init__(self, database):
        # A list of subsets of graph identifiers.
        # Is used to specify different groups of graphs (classes and training/test sets).
        # The gid-subsets parameter in the pruning and store function will contain for each subset, all the occurrences
        # in which the examined pattern is present.
        self.gid_subsets = []

        self.database = database  # A graphdatabase instance: contains the data for the problem.

    def store(self, dfs_code, gid_subsets):
        """
        Code to be executed to store the pattern, if desired.
        The function will only be called for patterns that have not been pruned.
        In correlated pattern mining, we may prune based on confidence, but then check further conditions before storing.
        :param dfs_code: the dfs code of the pattern (as a string).
        :param gid_subsets: the cover (set of graph ids in which the pattern is present) for each subset in self.gid_subsets
        """
        print(
            "Please implement the store function in a subclass for a specific mining task!"
        )

    def prune(self, gid_subsets):
        """
        prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
        should be pruned.
        :param gid_subsets: A list of the cover of the pattern for each subset.
        :return: true if the pattern should be pruned, false otherwise.
        """
        print(
            "Please implement the prune function in a subclass for a specific mining task!"
        )


class FrequentPositiveGraphs(PatternGraphs):
    """
    Finds the frequent (support >= minsup) subgraphs among the positive graphs.
    This class provides a method to build a feature matrix for each subset.
    """
    def __init__(self, minsup, database, subsets, top_K):
        """
        Initialize the task.
        :param minsup: the minimum positive support
        :param database: the graph database
        :param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
        """
        super().__init__(database)
        self.patterns = [
        ]  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
        self.minsup = minsup
        self.gid_subsets = subsets
        #self.gid_dubsets = sebset_test
        self.top_list = []  # list with all top items
        self.top_K = top_K
        self.top_current = 0  # current number of top items

    """ remove all elements that have c as conference and f as frequence. """
    def remove(self, c, f):
        self.patterns = [
            item for item in self.patterns if item[0] != c or item[1] != f
        ]
        self.top_list = [
            item for item in self.top_list if item[0] != c or item[1] != f
        ]

    """ delete the least frequent items with value c as confedence. """
    def delete(self, c):

        minF = float("inf")
        i = 0
        while (i < len(self.top_list) and self.top_list[i][0] == c):
            if (minF > self.top_list[i][1]):
                minF = self.top_list[i][1]
            i += 1

        #print("je suis virÃ©")
        #print(minF)
        #print(c)
        self.remove(c, minF)

    """ check if there are any items with confidence c and frequence f. """
    def check_eq_freq(self, c, f):
        i = 0
        while (i < len(self.top_list)):
            if (self.top_list[i][0] == c and self.top_list[i][1] == f):
                return True
            i += 1
        return False

    """ check if there are any items with confidence c but their frequence is smaller than f. """
    def check_min_freq(self, c, f):
        i = 0
        while (i < len(self.top_list)):
            if (self.top_list[i][0] == c and self.top_list[i][1] < f):
                return True
            i += 1
        return False

    """ Stores top_K patterns. """
    def store(self, dfs_code, gid_subsets):
        p = len(gid_subsets[0])
        n = len(gid_subsets[2])
        t = p + n
        pos_c = p / t
        neg_c = n / t
        if pos_c >= neg_c:
            c = pos_c
            is_pos = True
        else:
            c = neg_c
            is_pos = False
        if (self.check_eq_freq(c, t)):
            insort(self.patterns, [c, t, dfs_code, gid_subsets, is_pos])
            insort(self.top_list, [c, t])
        elif (self.top_current < self.top_K):
            insort(self.patterns, [c, t, dfs_code, gid_subsets, is_pos])
            insort(self.top_list, [c, t])
            self.top_current += 1
        elif (self.top_current == self.top_K):
            if (self.patterns[0][0] < c):
                self.delete(self.patterns[0][0])
                insort(self.patterns, [c, t, dfs_code, gid_subsets, is_pos])
                insort(self.top_list, [c, t])
            elif (self.patterns[0][0] == c and self.check_min_freq(c, t)):
                self.delete(self.patterns[0][0])
                insort(self.patterns, [c, t, dfs_code, gid_subsets, is_pos])
                insort(self.top_list, [c, t])

    """ Prunes any pattern that is not frequent. """
    def prune(self, gid_subsets):
        # first subset is the set of positive ids
        p = len(gid_subsets[0])
        n = len(gid_subsets[2])
        t = p + n
        return t < self.minsup


def task3(database_file_name_pos, database_file_name_neg, top_K, minsup, nfolds):
    """
    Runs gSpan with the specified positive and negative graphs; finds all frequent subgraphs in the training subset of
    the positive class with a minimum support of minsup.
    Uses the patterns found to train a naive bayesian classifier using Scikit-learn and evaluates its performances on
    the test set.
    Performs a k-fold cross-validation.
    """


    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(
        database_file_name_pos
    )  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(
        database_file_name_neg
    )  # Reading negative graphs, adding them to database and getting ids

    # If less than two folds: using the same set as training and test set (note this is not an accurate way to evaluate the performances!)
    if nfolds < 2:
        subsets = [
            pos_ids,  # Positive training set
            pos_ids,  # Positive test set
            neg_ids,  # Negative training set
            neg_ids  # Negative test set
        ]
        # Printing fold number:
        print('fold {}'.format(1))
        train_and_evaluate(minsup, graph_database, subsets, top_K)

    # Otherwise: performs k-fold cross-validation:
    else:
        pos_fold_size = len(pos_ids) // nfolds
        neg_fold_size = len(neg_ids) // nfolds
        for i in range(nfolds):
            # Use fold as test set, the others as training set for each class;
            # identify all the subsets to be maintained by the graph mining algorithm.
            subsets = [
                numpy.concatenate(
                    (pos_ids[:i * pos_fold_size],
                     pos_ids[(i + 1) *
                             pos_fold_size:])),  # Positive training set
                pos_ids[i * pos_fold_size:(i + 1) *
                        pos_fold_size],  # Positive test set
                numpy.concatenate(
                    (neg_ids[:i * neg_fold_size],
                     neg_ids[(i + 1) *
                             neg_fold_size:])),  # Negative training set
                neg_ids[i * neg_fold_size:(i + 1) *
                        neg_fold_size],  # Negative test set
            ]
            # Printing fold number:
            print('fold {}'.format(i + 1))
            train_and_evaluate(minsup, graph_database, subsets, top_K)


"""remove list1's elements from list 2"""


def remove(list1, list2):
    for elem in range(len(list1)):
        list_1, list_2 = list1[elem], list2[elem]
        for item in list_1:
            list_2.remove(item)
    return list2


def train_and_evaluate(minsup, database, subsets, top_K):

    pos_ids = copy.deepcopy(subsets[1])
    neg_ids = copy.deepcopy(subsets[3])
    new_subsets = []
    for subset in subsets:
        if type(subset) != type([]):
            new_subset = subset.tolist()
            new_subsets.append(new_subset)
        else:
            new_subsets.append(subset)

    result = []
    test_is_pos = []
    for i in range(top_K):
        task = FrequentPositiveGraphs(minsup, database, new_subsets, 1)
        gSpan(task).run()
        sort_list = []
        for pattern in task.patterns:
            sort_list.append([pattern[2], pattern[0], pattern[1], pattern[3]])
        sort_list.sort()
        if len(sort_list) > 0:
            result.append(sort_list[0])
            subsets_list = sort_list[0][3]
            test_list = subsets_list[1] + subsets_list[3]

            for item in test_list:
                insort(test_is_pos, [item, pattern[4]])

            new_subsets = remove(subsets_list, new_subsets)

    test_list = new_subsets[1] + new_subsets[3]
    length_pos = len(new_subsets[0])
    length_neg = len(new_subsets[2])

    if length_pos >= length_neg:
        is_pos = True
    else:
        is_pos = False

    for item in test_list:
        insort(test_is_pos, [item, is_pos])

    for pattern in result:
        print('{} {} {}'.format(pattern[0], pattern[1], pattern[2]))

    pred_result = []
    for pred in test_is_pos:
        if pred[1]:
            pred_result.append(1)
        else:
            pred_result.append(-1)

    print(pred_result)

    counter = 0

    for is_pos in test_is_pos:
        if is_pos[0] in pos_ids:
            if is_pos[1]:
                counter += 1
        if is_pos[0] in neg_ids:
            if not is_pos[1]:
                counter += 1
    accuracy = counter / len(test_is_pos)
    print('accuracy: {}'.format(accuracy))
    print()


def tae(minsup, database, subsets, top_K):

    pos_ids = copy.deepcopy(subsets[1])
    neg_ids = copy.deepcopy(subsets[3])

    pos_ids2 = copy.deepcopy(subsets[0])
    neg_ids2 = copy.deepcopy(subsets[2])

    new_subsets = []
    for subset in subsets:
        if type(subset) != type([]):
            new_subset = subset.tolist()
            new_subsets.append(new_subset)
        else:
            new_subsets.append(subset)

    result = []
    test_is_pos = []
    train_is_pos = []
    for i in range(top_K):
        task = FrequentPositiveGraphs(minsup, database, new_subsets, 1)
        gSpan(task).run()
        sort_list = []
        for pattern in task.patterns:
            sort_list.append([pattern[2], pattern[0], pattern[1], pattern[3]])
        sort_list.sort()
        if len(sort_list) > 0:
            result.append(sort_list[0])
            subsets_list = sort_list[0][3]
            test_list = subsets_list[1] + subsets_list[3]

            train_list = subsets_list[0] + subsets_list[2]

            for item in test_list:
                insort(test_is_pos,
                       [item, pattern[4]])  # pattern[4]: pos or not

            for item in train_list:
                insort(train_is_pos, [item, pattern[4]])

            new_subsets = remove(subsets_list, new_subsets)

    test_list = new_subsets[1] + new_subsets[3]
    train_list = new_subsets[0] + new_subsets[2]
    length_pos = len(new_subsets[0])
    length_neg = len(new_subsets[2])

    if length_pos >= length_neg:
        is_pos = True
    else:
        is_pos = False

    for item in test_list:
        insort(test_is_pos, [item, is_pos])

    for item in train_list:
        insort(train_is_pos, [item, is_pos])

    counter = 0

    for is_pos in test_is_pos:
        if is_pos[0] in pos_ids:
            if is_pos[1]:
                counter += 1
        if is_pos[0] in neg_ids:
            if not is_pos[1]:
                counter += 1
    testaccuracy = counter / len(test_is_pos)

    counter = 0

    for is_pos in train_is_pos:
        if is_pos[0] in pos_ids2:
            if is_pos[1]:
                counter += 1
        if is_pos[0] in neg_ids2:
            if not is_pos[1]:
                counter += 1
    trainaccuracy = counter / len(train_is_pos)

    return testaccuracy, trainaccuracy


def benchmark(posf, negf, nfolds, minsup, top_K):
    """
    Runs gSpan with the specified positive and negative graphs; finds all frequent subgraphs in the training subset of
    the positive class with a minimum support of minsup.
    Uses the patterns found to train a classifier using Scikit-learn and evaluates its performances on
    the test set.
    Performs a k-fold cross-validation.
    """

    prefix = "../statement/data/"

    database_file_name_pos = prefix + posf
    database_file_name_neg = prefix + negf

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(
        database_file_name_pos
    )  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(
        database_file_name_neg
    )  # Reading negative graphs, adding them to database and getting ids

    pos_fold_size = len(pos_ids) // nfolds
    neg_fold_size = len(neg_ids) // nfolds

    accuracy = {'test': [], 'train': []}

    for i in range(nfolds):
        # Use fold as test set, the others as training set for each class;
        # identify all the subsets to be maintained by the graph mining algorithm.
        subsets = [
            numpy.concatenate(
                (pos_ids[:i * pos_fold_size],
                 pos_ids[(i + 1) * pos_fold_size:])),  # Positive training set
            pos_ids[i * pos_fold_size:(i + 1) *
                    pos_fold_size],  # Positive test set
            numpy.concatenate(
                (neg_ids[:i * neg_fold_size],
                 neg_ids[(i + 1) * neg_fold_size:])),  # Negative training set
            neg_ids[i * neg_fold_size:(i + 1) *
                    neg_fold_size],  # Negative test set
        ]
        testacc, trainacc = tae(minsup, graph_database, subsets, top_K)
        accuracy['test'].append(testacc)
        accuracy['train'].append(trainacc)

    return accuracy



if __name__ == '__main__':

    database_file_name_pos = "/mnt/c/Users/b_tib/coding/Msc/oLING2364/Assignements/graph-mining/data/moleculesspos.txt"
    database_file_name_neg = "/mnt/c/Users/b_tib/coding/Msc/oLING2364/Assignements/graph-mining/data/moleculessneg.txt"
    top_K = 15  # Third parameter: k
    minsup = 2  # Fourth parameter: minimum support
    nfolds = 5

    task3(database_file_name_pos, database_file_name_neg, top_K, minsup, nfolds)
