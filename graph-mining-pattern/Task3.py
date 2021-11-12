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
    def __init__(self, minsup, database, subsets, k):
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
        self.rank = []  # list with all top items
        self.k = k
        self.rank_curr = 0  # current number of top items

    """ remove all elements that have c as conference and f as frequence. """
    def delete(self, conf):
        minF = float("inf")
        i = 0
        while (i < len(self.rank) and self.rank[i][0] == conf):
            if (minF > self.rank[i][1]):
                minF = self.rank[i][1]
            i += 1
        self.patterns = [item for item in self.patterns if item[0] != conf or item[1] != minF]
        self.rank = [item for item in self.rank if item[0] != conf or item[1] != minF]



    """ check if there are any items with confidence c and frequence f. """
    def match_freq(self, conf, freq):
        for i in range(len(self.rank)):
            if (self.rank[i][0] == conf and self.rank[i][1] == freq):
                return True
        return False

    """ check if there are any items with confidence c but their frequence is smaller than f. """
    def is_frequent(self, conf, total):
        i = 0
        for i in range(len(self.rank)):
            if (self.rank[i][0] == conf and self.rank[i][1] < total):
                return True
        return False

    """ Stores k patterns. """
    def store(self, dfs_code, gid_subsets):

        total = len(gid_subsets[0])+ len(gid_subsets[2])
        pos_conf = True
        conf = len(gid_subsets[0]) / total
        if len(gid_subsets[2]) > len(gid_subsets[0]):
            pos_conf = False
            conf = len(gid_subsets[2]) / total

        if (self.match_freq(conf, total)):
            insort(self.patterns, [conf, total, dfs_code, gid_subsets, pos_conf])
            insort(self.rank, [conf, total])
        elif (self.rank_curr < self.k):
            insort(self.patterns, [conf, total, dfs_code, gid_subsets, pos_conf])
            insort(self.rank, [conf, total])
            self.rank_curr += 1
        elif (self.rank_curr == self.k):
            if (self.patterns[0][0] < conf):
                self.delete(self.patterns[0][0])
                insort(self.patterns, [conf, total, dfs_code, gid_subsets, pos_conf])
                insort(self.rank, [conf, total])
            elif (self.patterns[0][0] == conf and self.is_frequent(conf, total)):
                self.delete(self.patterns[0][0])
                insort(self.patterns, [conf, total, dfs_code, gid_subsets, pos_conf])
                insort(self.rank, [conf, total])

    """ Prunes any pattern that is not frequent. """
    def prune(self, gid_subsets):
        # first subset is the set of positive ids
        total = len(gid_subsets[0]) + len(gid_subsets[2])
        return total < self.minsup


def task3(database_file_name_pos, database_file_name_neg, k, minsup, nfolds):
    """
    Runs gSpan with the specified positive and negative graphs;
    Encounter frequents and train the model;
    Uses the patterns found to train a Decision Tree classifier  and validates using k-fold cross-validation.
    """

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

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
        train_and_evaluate(minsup, graph_database, subsets, k)

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
            train_and_evaluate(minsup, graph_database, subsets, k)



def train_and_evaluate(minsup, database, subsets, k):

    pos_ids = copy.deepcopy(subsets[1])
    neg_ids = copy.deepcopy(subsets[3])


    list_subsets = []
    for subset in subsets:
        if isinstance(subset, list):
            list_subsets.append(subset)
        else:
            ready_to_go = subset.tolist()
            list_subsets.append(ready_to_go)

    result = []
    temp_conf = []
    for i in range(k):
        task = FrequentPositiveGraphs(minsup, database, list_subsets, 1)
        gSpan(task).run()
        sorted_list = []
        for pattern in task.patterns:
            sorted_list.append([pattern[2], pattern[0], pattern[1], pattern[3]])
        sorted_list.sort()
        if len(sorted_list) > 0:
            result.append(sorted_list[0])
            subsets_list = sorted_list[0][3]
            test_list = subsets_list[1] + subsets_list[3]

            list_subsets = [[x for x in b if x not in a] for a, b in zip(subsets_list, list_subsets)]

            for item in test_list:
                insort(temp_conf, [item, pattern[4]])

    test_list = list_subsets[1] + list_subsets[3]

    pos_conf = True
    if len(list_subsets[0]) < len(list_subsets[2]):
        pos_conf = False

    for item in test_list:
        insort(temp_conf, [item, pos_conf])

    for pattern in result:
        print('{} {} {}'.format(pattern[0], pattern[1], pattern[2]))

    pred_result = []
    for pred in temp_conf:
        if pred[1]:
            pred_result.append(1)
        else:
            pred_result.append(-1)
    print(pred_result)

    counter = 0
    for pos_conf in temp_conf:
        if pos_conf[0] in pos_ids:
            if pos_conf[1]:
                counter += 1
        if pos_conf[0] in neg_ids:
            if not pos_conf[1]:
                counter += 1
    accuracy = counter / len(temp_conf)
    print('accuracy: {}'.format(accuracy))
    print()


def tae(minsup, database, subsets, k):

    pos_ids = copy.deepcopy(subsets[1])
    neg_ids = copy.deepcopy(subsets[3])

    pos_ids2 = copy.deepcopy(subsets[0])
    neg_ids2 = copy.deepcopy(subsets[2])

    list_subsets = []
    for subset in subsets:
        if type(subset) != type([]):
            new_subset = subset.tolist()
            list_subsets.append(new_subset)
        else:
            list_subsets.append(subset)

    result = []
    temp_conf = []
    train_pos_conf = []
    for i in range(k):
        task = FrequentPositiveGraphs(minsup, database, list_subsets, 1)
        gSpan(task).run()
        sorted_list = []
        for pattern in task.patterns:
            sorted_list.append([pattern[2], pattern[0], pattern[1], pattern[3]])
        sorted_list.sort()
        if len(sorted_list) > 0:
            result.append(sorted_list[0])
            subsets_list = sorted_list[0][3]
            test_list = subsets_list[1] + subsets_list[3]

            train_list = subsets_list[0] + subsets_list[2]

            for item in test_list:
                insort(temp_conf,
                       [item, pattern[4]])
            for item in train_list:
                insort(train_pos_conf, [item, pattern[4]])

            list_subsets = [[x for x in b if x not in a] for a, b in zip(subsets_list, list_subsets)]

    test_list = list_subsets[1] + list_subsets[3]
    train_list = list_subsets[0] + list_subsets[2]

    pos_conf = True
    if len(list_subsets[0]) < len(list_subsets[2]):
        pos_conf = False

    # building test and training lists with conf, item & boolean
    for item in test_list:
        insort(temp_conf, [item, pos_conf])
    for item in train_list:
        insort(train_pos_conf, [item, pos_conf])

    # test accuracy
    counter = 0
    for pos_conf in temp_conf:
        if pos_conf[0] in pos_ids:
            if pos_conf[1]:
                counter += 1
        if pos_conf[0] in neg_ids:
            if not pos_conf[1]:
                counter += 1
    testaccuracy = counter / len(temp_conf)

    # training accuracy
    counter = 0
    for pos_conf in train_pos_conf:
        if pos_conf[0] in pos_ids2:
            if pos_conf[1]:
                counter += 1
        if pos_conf[0] in neg_ids2:
            if not pos_conf[1]:
                counter += 1
    trainaccuracy = counter / len(train_pos_conf)

    return testaccuracy, trainaccuracy



#################
# # Run Local
# #################


if __name__ == '__main__':

    database_file_name_pos = "/mnt/c/Users/b_tib/coding/Msc/oLING2364/Assignements/graph-mining/data/moleculesspos.txt"
    database_file_name_neg = "/mnt/c/Users/b_tib/coding/Msc/oLING2364/Assignements/graph-mining/data/moleculessneg.txt"
    k = 15  # Third parameter: k
    minsup = 30  # Fourth parameter: minimum support
    nfolds = 5

    task3(database_file_name_pos, database_file_name_neg, k, minsup, nfolds)


# ###############
# #Run Inginious
# ###############

# if __name__ == '__main__':
#     args = sys.argv
#     database_file_name_pos = args[1]  # First parameter: path to positive class file
#     database_file_name_neg = args[2]  # Second parameter: path to negative class file
#     k = int(args[3])  # Third parameter: k
#     minsup = int(args[4])  # Fourth parameter: minimum support
#     nfolds = int(args[5])  # Fifth parameter: number of folds to use in the k-fold cross-validation.



#     task3(database_file_name_pos, database_file_name_neg, k, minsup, nfolds)
