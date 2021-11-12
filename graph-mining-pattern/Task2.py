"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy
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
        self.rank = []
        self.k = k
        self.rank_curr = 0


    '''Remove least frequent from Rank with a confidence==conf'''
    def delete(self, conf):
        minF = float("inf")
        i = 0
        while (i < len(self.rank) and self.rank[i][0] == conf):
            if (minF > self.rank[i][1]):
                minF = self.rank[i][1]
            i += 1
        self.patterns = [item for item in self.patterns if item[0] != conf or item[1] != minF]
        self.rank = [item for item in self.rank if item[0] != conf or item[1] != minF]


    '''check if conf and freq repeats'''
    def match_freq(self, conf, freq):
        for i in range(len(self.rank)):
            if (self.rank[i][0] == conf and self.rank[i][1] == freq):
                return True
        return False

    ''' check confidence confidence==conf and their frequence is smaller than total. '''
    def is_frequent(self, conf, total):
        i = 0
        for i in range(len(self.rank)):
            if (self.rank[i][0] == conf and self.rank[i][1] < total):
                return True
        return False

    """ Stores k patterns. """
    def store(self, dfs_code, gid_subsets):

        total = len(gid_subsets[0])+ len(gid_subsets[2])
        conf = len(gid_subsets[0]) / total
        if (self.match_freq(conf, total)):
            insort(self.patterns, [conf, total, dfs_code, gid_subsets])
            insort(self.rank, [conf, total])
        elif (self.rank_curr < self.k):
            insort(self.patterns, [conf, total, dfs_code, gid_subsets])
            insort(self.rank, [conf, total])
            self.rank_curr += 1
        elif (self.rank_curr == self.k):
            if (self.patterns[0][0] < conf):
                self.delete(self.patterns[0][0])
                insort(self.patterns, [conf, total, dfs_code, gid_subsets])
                insort(self.rank, [conf, total])
            elif (self.patterns[0][0] == conf and self.is_frequent(conf, total)):
                self.delete(self.patterns[0][0])
                insort(self.patterns, [conf, total, dfs_code, gid_subsets])
                insort(self.rank, [conf, total])

    """ Prunes any pattern that is not frequent. """
    def prune(self, gid_subsets):
        # first subset is the set of positive ids
        total = len(gid_subsets[0]) + len(gid_subsets[2])
        return total < self.minsup

    """ creates a column for a feature matrix. """
    def create_fm_col(self, all_gids, subset_gids):
        subset_gids = set(subset_gids)
        bools = []
        for i, val in enumerate(all_gids):
            if val in subset_gids:
                bools.append(1)
            else:
                bools.append(0)
        return bools

    """ return a feature matrix for each subset of examples, in which the columns correspond to patterns
         and the rows to examples in the subset. """
    def get_feature_matrices(self):
        matrices = [[] for _ in self.gid_subsets]
        for pattern in self.patterns:
            for i, gid_subset in enumerate(pattern[3]):
                matrices[i].append(
                    self.create_fm_col(self.gid_subsets[i], gid_subset))
        return [numpy.array(matrix).transpose() for matrix in matrices]


def task2(database_file_name_pos, database_file_name_neg, k, minsup, nfolds):
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
    task = FrequentPositiveGraphs(minsup, database, subsets, k)  # Creating task


    gSpan(task).run()  # Running gSpan

    # Creating feature matrices for training and testing:
    features = task.get_feature_matrices()
    # print('features',features)
    train_fm = numpy.concatenate(
        (features[0], features[2]))  # Training feature matrix
    train_labels = numpy.concatenate(
        (numpy.full(len(features[0]), 1,
                    dtype=int), numpy.full(len(features[2]), -1,
                                           dtype=int)))  # Training labels
    test_fm = numpy.concatenate(
        (features[1], features[3]))  # Testing feature matrix
    test_labels = numpy.concatenate((numpy.full(len(features[1]), 1, dtype=int), numpy.full(len(features[3]), -1,
                                           dtype=int)))  # Testing labels

    classifier = tree.DecisionTreeClassifier(random_state=1)  # Creating model object
    classifier.fit(train_fm, train_labels)  # Training model

    predictedtest = classifier.predict(test_fm)  # Using model to predict labels of testing data

    testaccuracy = metrics.accuracy_score(test_labels, predictedtest)  # Computing accuracy:

	# Printing frequent patterns along with their positive support:
    for pattern in task.patterns:
        total_support = pattern[1]
        confidence = pattern[0]
        print('{} {} {}'.format(pattern[2], confidence, total_support))
        # printing classification results:
    print(predictedtest.tolist())
    print('accuracy: {}'.format(testaccuracy))
    print()  # Blank line to indicate end of fold.


# #################
# # Run Local
# #################


# if __name__ == '__main__':

#     database_file_name_pos = "/mnt/c/Users/b_tib/coding/Msc/oLING2364/Assignements/graph-mining/data/moleculesspos.txt"
#     database_file_name_neg = "/mnt/c/Users/b_tib/coding/Msc/oLING2364/Assignements/graph-mining/data/moleculessneg.txt"
#     k = 5  # Third parameter: k
#     minsup = 5  # Fourth parameter: minimum support
#     nfolds = 4

#     task2(database_file_name_pos, database_file_name_neg, k, minsup, nfolds)


################
#Run Inginious
################

if __name__ == '__main__':
    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    k = int(args[3])  # Third parameter: k
    minsup = int(args[4])  # Fourth parameter: minimum support
    nfolds = int(args[5])  # Fifth parameter: number of folds to use in the k-fold cross-validation.
    task2(database_file_name_pos, database_file_name_neg, k, minsup, nfolds)
