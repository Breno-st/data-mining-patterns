


"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from gspan_mining import gSpan
from gspan_mining import GraphDatabase
import statistics

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
        print("Please implement the store function in a subclass for a specific mining task!")

    def prune(self, gid_subsets):
        """
        prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
        should be pruned.
        :param gid_subsets: A list of the cover of the pattern for each subset.
        :return: true if the pattern should be pruned, false otherwise.
        """
        print("Please implement the prune function in a subclass for a specific mining task!")


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
        self.patterns = []  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
        self.minsup = minsup
        self.gid_subsets = subsets
        self.k = k
        self.confidence_frequence = set()

    # Stores any pattern found that has not been pruned
    def store(self, dfs_code, gid_subsets):
        self.patterns.append((dfs_code, gid_subsets))

    def sortPatters(self):
        newPattern = []
        counter = 0
        bound = 1.01
        boundFreq = len(self.patterns) +1
        while counter < self.k :
            pat = self.maxconf(bound)
            patfreq = self.maxFreq(pat,boundFreq)
            while(len(patfreq)> 0 and counter < self.k):
                counter = counter + 1
                pos = patfreq[0][1]
                neg = patfreq[0][2]
                bound = pos / (pos+neg)
                #bound = bound - 10e-18
                for elem in patfreq:
                    newPattern.append((elem[0],elem[3]))
                freq = pos + neg
                patfreq = self.maxFreq(pat,freq)
        self.patterns = newPattern

    def maxFreq(self, pat, boundfreq):
        biggest = 0
        result = list()
        for pattern in pat:
            if (pattern[1] + pattern[2]) == biggest:
                result.append(pattern)

            if (pattern[1] + pattern[2]) > biggest and (pattern[1] + pattern[2]) < boundfreq:
                result = [pattern]
                biggest = pattern[1] + pattern[2]

        return result


    def maxconf(self, bound):
        pat = list() # elements of pattern,pos,neg,confidence
        conf = 0
        for pattern, gid_subsets in self.patterns:
            pos_support = len(gid_subsets[0])
            neg_support = len(gid_subsets[2])
            confidence = pos_support / (pos_support + neg_support)

            if(confidence == conf):
                pat.append([pattern,pos_support,neg_support,gid_subsets])

            if(confidence > conf and confidence < bound):
                pat = [[pattern,pos_support,neg_support,gid_subsets]]
                conf = confidence

        return pat

    # Prunes any pattern that is not frequent in the positive class
    def prune(self, gid_subsets):
        # first subset is the set of positive ids
        return (len(gid_subsets[0]) + len(gid_subsets[2])) < self.minsup

    # creates a column for a feature matrix
    def create_fm_col(self, all_gids, subset_gids):
        subset_gids = set(subset_gids)
        bools = []
        for i, val in enumerate(all_gids):
            if val in subset_gids:
                bools.append(1)
            else:
                bools.append(0)
        return bools

    # return a feature matrix for each subset of examples, in which the columns correspond to patterns
    # and the rows to examples in the subset.
    def get_feature_matrices(self):
        matrices = [[] for _ in self.gid_subsets]
        count = 0
        for pattern, gid_subsets in self.patterns:
            for i, gid_subset in enumerate(gid_subsets):
                matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))

            count = count+1
        return [numpy.array(matrix).transpose() for matrix in matrices]


def topK(database_file_name_pos, database_file_name_neg ,k, minsup, nfolds):
    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids
    #print(graph_database._graphs[0].plot())



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
        train_and_evaluate(minsup, graph_database, subsets)

    # Otherwise: performs k-fold cross-validation:
    else:
        pos_fold_size = len(pos_ids) // nfolds
        neg_fold_size = len(neg_ids) // nfolds
        for i in range(nfolds):
            # Use fold as test set, the others as training set for each class;
            # identify all the subsets to be maintained by the graph mining algorithm.
            subsets = [
                numpy.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),  # Positive training set
                pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
                numpy.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),  # Negative training set
                neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
            ]
            # Printing fold number:
            print('fold {}'.format(i+1))
            train_and_evaluate(minsup, graph_database, subsets, k)



def train_and_evaluate(minsup, database, subsets, k):
    task = FrequentPositiveGraphs(minsup, database, subsets, k)  # Creating task
    gSpan(task).run()  # Running gSpan

    task.sortPatters()







    features = task.get_feature_matrices()
    train_fm = numpy.concatenate((features[0], features[2]))  # Training feature matrix
    train_labels = numpy.concatenate((numpy.full(len(features[0]), 1, dtype=int), numpy.full(len(features[2]), -1, dtype=int)))  # Training labels
    test_fm = numpy.concatenate((features[1], features[3]))  # Testing feature matrix
    test_labels = numpy.concatenate((numpy.full(len(features[1]), 1, dtype=int), numpy.full(len(features[3]), -1, dtype=int)))  # Testing labels

    #classifier = naive_bayes.GaussianNB()
    classifier = svm.SVC() 
#    classifier = KNeighborsClassifier(n_neighbors=6)
    classifier.fit(train_fm, train_labels)  # Training model



    predicted = classifier.predict(test_fm)  # Using model to predict labels of testing data

    accuracy = metrics.accuracy_score(test_labels, predicted)  # Computing accuracy:
    for pattern, gid_subsets in task.patterns:
        print(' {} {} {}'.format(pattern,(len(gid_subsets[0]) / (len(gid_subsets[0])+len(gid_subsets[2]))),(len(gid_subsets[0])+len(gid_subsets[2]))))


    print(predicted.tolist())
    print('accuracy: {}'.format(accuracy))
    print()  # Blank line to indicate end of fold


if __name__ == '__main__':
     args = sys.argv
     database_file_name_pos = args[1]  # First parameter: path to positive class file
     database_file_name_neg = args[2]  # Second parameter: path to negative class file
     nfolds = int(args[3])  # Fifth parameter: number of folds to use in the k-fold cross-validation.
     topK(database_file_name_pos, database_file_name_neg, 4, 10, nfolds)