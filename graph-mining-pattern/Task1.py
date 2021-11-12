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
    def __init__(self, database, subsets, minsup, k):
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
        self.rank = []  # list with all top items
        self.k = k
        self.rank_curr = 0  # current number of top items




    """ Stores rank patterns. """
    def store(self, dfs_code, gid_subsets):
        total = len(gid_subsets[0]) + len(gid_subsets[1])
        conf = len(gid_subsets[0]) / total
        if self.match_freq(conf, total):
            insort(self.patterns, [conf, total, dfs_code])
            insort(self.rank, [conf, total])
        elif self.rank_curr < self.k:
            insort(self.patterns, [conf, total, dfs_code])
            insort(self.rank, [conf, total])
            self.rank_curr += 1
        elif self.rank_curr == self.k:
            if self.patterns[0][0] < conf:
                self.delete(self.patterns[0][0])
                insort(self.patterns, [conf, total, dfs_code])
                insort(self.rank, [conf, total])
            elif self.patterns[0][0] == conf and self.is_frequent(conf, total):
                self.delete(self.patterns[0][0])
                insort(self.patterns, [conf, total, dfs_code])
                insort(self.rank, [conf, total])

    """ Prunes any pattern that is not frequent. """
    def prune(self, gid_subsets):
        # first subset is the set of positive ids
        total = len(gid_subsets[0]) + len(gid_subsets[1])
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
        for pattern, gid_subsets in self.patterns:
            for i, gid_subset in enumerate(gid_subsets):
                matrices[i].append(
                    self.create_fm_col(self.gid_subsets[i], gid_subset))
        return [numpy.array(matrix).transpose() for matrix in matrices]

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


def task1(database_file_name_pos, database_file_name_neg, k, minsup):
    """
    Runs gSpan with the specified positive and negative graphs, finds all topK frequent subgraphs based on their confidence
    with a minimum positive support of minsup and prints them.
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

    subsets = [pos_ids, neg_ids]  # The ids for the positive and negative labelled graphs in the database
    task = FrequentPositiveGraphs(graph_database, subsets, minsup, k)  # Creating task

    gSpan(task).run()  # Running gSpan

    # Printing frequent patterns along with their confidence and total support:
    for pattern in task.patterns:
        total_support = pattern[1]
        confidence = pattern[0]
        print('{} {} {}'.format(pattern[2], confidence, total_support))


#################
# Run Local
#################

if __name__ == '__main__':
	database_file_name_pos = "/mnt/c/Users/b_tib/coding/Msc/oLING2364/Assignements/graph-mining/data/moleculesspos.txt"
	database_file_name_neg = "/mnt/c/Users/b_tib/coding/Msc/oLING2364/Assignements/graph-mining/data/moleculessneg.txt"
	k = 5
	minsup = 5
	task1(database_file_name_pos, database_file_name_neg, k, minsup)


# #################
# # Inginious
# #################
# if __name__ == '__main__':
# 	args = sys.argv
# 	database_file_name_pos = args[1]  # First parameter: path to positive class file
# 	database_file_name_neg = args[2]  # Second parameter: path to negative class file
# 	k = int(args[3])  # Third parameter: k
# 	minsup = int(args[4])  # Fourth parameter: minimum support
# 	task1(database_file_name_pos, database_file_name_neg, k, minsup)
