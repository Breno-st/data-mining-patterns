# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Tue Apr 12 09:21:49 2021

@author: breno.tiburcio@student.uclouvain.be
"""
import sys
import numpy as np
from collections import Counter
import copy


class Dataset:
    '''Manage classes pathfiles and consolidated dataset'''
    def __init__(self, *args):
        n = len(args)
        self.trans = list()
        self.items = set()
        if n == 1 and isinstance(args[0], str):
            self.get_class(args[0])
        else:
            for i in range(n-1):
                if not isinstance(args[i], Dataset):
                    sys.exit('Not possible to merge dataset. Check if entry No.'+ str(i+1) + ' is Dataset object')
            self.get_data(args)

    # access filepath and convert it into a class dataset
    def get_class(self, filepath):
        try:
            lines = [line.strip() for line in open(filepath, "r")]
            listItem = list()
            for line in range(1,len(lines)):
                if(lines[line] == ''):
                    self.trans.append(listItem)
                    listItem = list()
                else:
                    transaction = list(lines[line].split(" "))
                    listItem.append(transaction[0])
                    self.items.add(transaction[0])
            self.len = len(self.trans)
            self.items = sorted(self.items)

        except IOError as e:
            print("Unable to read dataset file!\n" + e)

    # merge classes into a unique dataset
    def get_data (self, *args):
        self.cls = len(list(args[0]))
        self.cl = {}
        for idx, arg in enumerate(list(args[0])):
            self.cl[idx+1]={}
            self.cl[idx+1]['t'] = arg.trans
            self.cl[idx+1]['i'] = arg.items
            self.cl[idx+1]['l'] = len(arg.trans)
            for trans in arg.trans:
                self.trans.append(trans)
            for item in arg.items:
                self.items.add(item)
        self.len = len(self.trans)

# GenF 1
def prefixSpan(data, prefix):
    ''' Assemble prefix global variables to control recursion calls '''
    # Ordely get covers dictionaire for overall data and its classes.
    covers = cover(data) # GenF 2

    # interact overall items (covers[0])
    for item, recurr in covers[0].items():
        prefix.append(item)

        temp_prefix = copy.deepcopy(prefix)
        result_prefix[0].append(temp_prefix)
        result_prefix_recur[0].append(recurr)

        # update recurrence for classes
        for n in range(1, data.cls + 1):
            if item in covers[n]:
                result_prefix[n].append(temp_prefix)
                result_prefix_recur[n].append(covers[n][item])

        # recursion trigger
        if len(prefix) < 4:
            new_data = proj_db(data, item)
            prefixSpan(new_data, prefix)
        prefix.pop(len(prefix)-1)

# GenF 2
def cover(data):
    ''' Items overall recurrence covers[0]
        Items partial recurrence covers[n] '''
    covers = dict()
    for n in range(1, data.cls+1):
        covers[n]={}
        for trans in data.cl[n]['t']:
            for item in data.cl[n]['i']:
                if item in trans:
                    if item not in covers[n]: # if not in dic, add.
                        covers[n][item] = 1
                    else: # increment count.
                        covers[n][item] += 1

    counts = Counter()
    for k, v in covers.items():
         counts.update(v)

    cover = dict(sorted(counts.items(), key = lambda i: i[0]))
    covers[0] = cover

    # creates a list of recurrences
    item_recurr_lists = list()
    for item, recurr in covers[0].items():
        item_recurr_lists.append(item)

    # keeps cover dictionair updated
    for item in item_recurr_lists:
        if covers[0][item] < 2:
            del covers[0][item]

            for n in range(1, data.cls+1):
                if item in covers[n]:
                    del covers[n][item]
    return covers

# GenF 3
def get_recurr_prefix(result_prefix, result_prefix_recur, higher_recurr):

    recurr_prefix = list()
    for i in range(len(result_prefix)):
        if result_prefix_recur[i] == higher_recurr:
            recurr_prefix.append(result_prefix[i])
    return recurr_prefix

# GenF 4
def proj_db(data, item):
    ''' Items overall recurrence covers[0]
        Items partial recurrence covers[n] '''
    # Creates a new database
    new_data = copy.deepcopy(data)
    new_data.trans = list()
    temp_trans = list()
    for n in range(1, new_data.cls + 1):
        new_data.cl[n]['t'] = list()
        for trans in data.cl[n]['t']:
            count = 0
            while count < len(trans) and trans[count] != item:
                count += 1
            count += 1
            if(count < len(trans)):
                for pos in range(count, len(trans)):
                    temp_trans.append(trans[pos])

                new_data.trans.append(temp_trans)
                new_data.cl[n]['t'].append(temp_trans)
                temp_trans = list()

    return new_data

# The Sequence Mining Algo
def PrefixSpan(*args):
    global result_prefix
    global result_prefix_recur

    # Inginious workaround
    k = args[len(args) - 1]
    args = args[:-1]

    # setting results structure as dictionaires
    result_prefix = dict()
    result_prefix_recur = dict()

    # initiate the dataset based on available classes
    classes = list()
    for arg in args:
        classes.append(Dataset(arg))
    data = Dataset(*classes)

    # preparing results structure based on args
    for cls in range(len(args)+1):
        result_prefix[cls] = list()
        result_prefix_recur[cls] = list()

    # Initiate a global prefix list for recursions
    prefix = list()
    prefixSpan(data, prefix)

    result_prefix_ = dict()
    result_prefix_recur_ = dict()

    for cls in range(len(args)+1):
        result_prefix_[cls] = copy.deepcopy(result_prefix[cls])
        result_prefix_recur_[cls] = copy.deepcopy(result_prefix_recur[cls])

    # getting max recurrence values
    higher_recurr = np.amax(result_prefix_recur_[0])

    counter = 0
    while(counter < k and higher_recurr > 0):

        # Pour une occuence particulière, recupere la liste d'items avec cete même occurence
        freq_prefixes = get_recurr_prefix(result_prefix_[0], result_prefix_recur_[0], higher_recurr)

        if len(freq_prefixes) > 0:

            counter = counter + 1
            # Retrive each frequent item
            for freq_prefix in freq_prefixes:

                class_recurr = dict()
                for n in range(1, len(args)+1):
                    class_recurr[n] = 0
                    for i in range(len(result_prefix_recur_[n])):
                        if result_prefix_[n][i] == freq_prefix:
                            class_recurr[n] = result_prefix_recur_[n][i]
                            i = len(result_prefix_recur_[n])

                st = "["
                for i in range(len(freq_prefix)):
                    if i != len(freq_prefix)-1:
                        st = st + freq_prefix[i] +", "
                    else:
                        st = st + freq_prefix[i] +"] "

                for n in range(1, data.cls + 1):
                    st = st + str(class_recurr[n])+" "

                st = st + str(higher_recurr)
                print(st)

        higher_recurr -= 1

# ############
# # Inginious
# ############
# def main():
#     pos_filepath = sys.argv[1] # filepath to positive class file
#     neg_filepath = sys.argv[2] # filepath to negative class file
#     k = int(sys.argv[3])
#     # TODO: read the dataset files and call your miner to print the top k itemsets
#     PrefixSpan(pos_filepath, neg_filepath, k)


# if __name__ == "__main__":
#     main()

#########################
# Initiate Algo Locally
#########################
neg_filepath = "/mnt/c/Users/b_tib/coding/Msc/oLING2364/Assignements/sequence-mining-pattern/Datasets/Protein/SRC1521.txt"
pos_filepath = "/mnt/c/Users/b_tib/coding/Msc/oLING2364/Assignements/sequence-mining-pattern/Datasets/Protein/PKA_group15.txt"


k = 10
PrefixSpan(pos_filepath, neg_filepath, k)
