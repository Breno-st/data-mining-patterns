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
import math

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
def prefixSpan(data, prefix, close):
    ''' Assemble prefix global variables to control recursion calls '''

    if len(data.items) < 4:
        limit = 6
    else:
        limit = 3

    # Ordely get covers dictionaire for overall data and its classes.
    covers = cover(data, prefix, close) # GenF 2

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
        if len(prefix) < limit:
            new_data = proj_db(data, item)
            prefixSpan(new_data, prefix, recurr)
        else:
            result_prefix_closed.append(temp_prefix)
            result_prefix_recur_closed.append(close)

        prefix.pop(len(prefix)-1)

# GenF 2
def cover(data, prefix, close):
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

    flagClose = True

    if(len(data.items) < 4):
        limit = 1
    else:
        limit = 2

    # creates a list of recurrences
    item_recurr_lists = list()
    for item, recurr in covers[0].items():
        item_recurr_lists.append(item)

    # keeps cover dictionair updated
    for item in item_recurr_lists:
        if covers[0][item] < limit:
            del covers[0][item]

            for n in range(1, data.cls+1):
                if item in covers[n]:
                    del covers[n][item]
        else:
            if(covers[0][item] == close):
                flagClose = False

    if flagClose and len(prefix) > 0:
        pref = copy.deepcopy(prefix)
        result_prefix_closed.append(pref)
        result_prefix_recur_closed.append(close)

    return covers

# GenF 3
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

# GenF 4
def imp(x):
        if(x == 1 or x == 0):
            return 0
        else:
            result = x* math.log(x, 2) + (1 - x) * math.log(1 - x, 2)
            result = result*-1
            return  result

# GenF 5
def infoGain(data, prefix_closed, recurr_closed, prefix_pos, recurr_pos, prefix_neg, recurr_neg):
    p = data.cl[1]['l']
    n = data.cl[2]['l']

    infog_prefix = list()
    infog_recurr = list()

    ###########
    # REamke for classes later
    for i in range(len(recurr_closed)):
        supPositif = 0
        supNegatif = 0
        j = 0
        while j < len(recurr_pos):
            if(prefix_pos[j] == prefix_closed[i]):
                supPositif= recurr_pos[j]
                j = len(recurr_pos)
            j += 1
        j = 0

        while j < len(recurr_neg):
            if(prefix_neg[j] == prefix_closed[i]):
                supNegatif= recurr_neg[j]
                j = len(recurr_neg)
            j = j+1

        if(supPositif == 0 and supNegatif==0):
            return 0
        if(p == 0 and n == 0):
            return 0
    ##########

        gain = imp(p/(p+n))
        gain = gain - (((supPositif + supNegatif)/(p+n)) * imp(supPositif/(supPositif + supNegatif)))
        if(p != supPositif or n!= supNegatif):
            gain = gain - (((p + n - supPositif - supNegatif)/(p + n)) * imp((p - supPositif)/(p + n - supPositif- supNegatif)))

        infog_prefix.append(prefix_closed[i])
        infog_recurr.append(gain)


    return infog_prefix, infog_recurr

# GenF 6
def next_recurr(recurr_list, limit):
    higher_recurr = 0
    for i in range(len(recurr_list)):
        if(recurr_list[i] > 0):
            if recurr_list[i] > higher_recurr and recurr_list[i] < limit:
                higher_recurr = recurr_list[i]
        else:
            neg = recurr_list[i] *-1
            if neg> higher_recurr and neg< limit:
                higher_recurr = neg
    return higher_recurr

# GenF 7
def compare_lists(listA, listB):
        i, j, pos = 0, 0, 0
        while i < len(listA):
            check = False
            j = pos
            while j < len(listB):
                if listA[i] == listB[j] :
                    check = True
                    pos = j + 1
                    j = len(listB)
                j += 1
            if check == False:
                i = len(listA)
            i += 1

        if check:
            return True
        else:
            return False

# GenF 8
def max_abs(recurr_list):
        higher_recurr = 0
        recurr_list_ = copy.deepcopy(recurr_list)
        for i in range(len(recurr_list)):
            if recurr_list_[i] < 0:
                recurr_list_[i] = recurr_list_[i] * -1

        for i in range(len(recurr_list_)):
            if recurr_list_[i] > higher_recurr:
                higher_recurr = recurr_list_[i]
        return higher_recurr

# GenF 9
def get_recurr_prefix(prefix_list, recurr_list, higher_recurr):
    higher_prefix_list = list()
    for i in range(len(prefix_list)):
        if recurr_list[i] == higher_recurr:
            higher_prefix_list.append(prefix_list[i])

    return higher_prefix_list

# GenF 10
def closed_list(list): #old one
    dic = dict()
    for i in range(len(list)):
        dic[i] = False

    for i in range(len(list) - 1):
        for j in range(i+1, len(list)):
            if(len(list[i]) < len(list[j])):
                check = compare_lists(list[i] ,list[j])
                if(check):
                    dic[i] = True
            else:
                check = compare_lists(list[j], list[i])
                if(check):
                    dic[j] = True

    j = len(list)-1
    while j > -1:
        if(dic[j] == True):
            list.pop(j)
        j -= 1


# The Sequence Mining Algo
def PrefixSpan_Sup_Clos_Info_Gain(*args):
    global result_prefix
    global result_prefix_recur
    global result_prefix_closed
    global result_prefix_recur_closed

    # setting results structure as dictionaires
    result_prefix = dict()
    result_prefix_recur = dict()
    result_prefix_closed = list()
    result_prefix_recur_closed = list()


	# Inginious workaround
    k = args[len(args) - 1]
    args = args[:-1]


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
    prefixSpan(data, prefix, sys.maxsize)

    result_prefix_ = dict()
    result_prefix_recur_ = dict()

    for cls in range(len(args)+1):
        result_prefix_[cls] = copy.deepcopy(result_prefix[cls])
        result_prefix_recur_[cls] = copy.deepcopy(result_prefix_recur[cls])

    result_prefix_closed_ = copy.deepcopy(result_prefix_closed)
    result_prefix_recur_closed_ = copy.deepcopy(result_prefix_recur_closed)


    infog_prefix, infog_recurr = infoGain(data
                                                 , result_prefix_closed_
                                                 , result_prefix_recur_closed_
                                                 , result_prefix_[1]
                                                 , result_prefix_recur_[1]
                                                 , result_prefix_[2]
                                                 , result_prefix_recur_[2])

    # rounding list
    for i in range(len(result_prefix_recur_closed_)):
        result_prefix_recur_closed_[i] = round(result_prefix_recur_closed_[i],9)


    # getting max recurrence values
    higher_recurr = max_abs(result_prefix_recur_closed_)


    after_prefix_list = list()

    while(higher_recurr > 0):
        long_prefix_list = get_recurr_prefix(result_prefix_closed_, result_prefix_recur_closed_, higher_recurr)
        closed_list(long_prefix_list)
        for prefix in long_prefix_list:
            after_prefix_list.append(prefix)
        higher_recurr = next_recurr(result_prefix_recur_closed_, higher_recurr)


    higher_recurr = max(infog_recurr)


    counter = 0
    while(counter < k and higher_recurr > 0):

        long_prefix_list = get_recurr_prefix(infog_prefix, infog_recurr, higher_recurr)

        if len(long_prefix_list) > 0:

            counter = counter + 1
            # Retrive each frequent item
            for long_prefix in long_prefix_list:

                if long_prefix in after_prefix_list:

                    class_recurr = dict()
                    for n in range(1, len(args)+1):
                        class_recurr[n] = 0
                        for i in range(len(result_prefix_recur_[n])):
                            if result_prefix_[n][i] == long_prefix:
                                class_recurr[n] = result_prefix_recur_[n][i]

                    st = "["
                    for i in range(len(long_prefix)):
                        if i != len(long_prefix)-1:
                            st = st + long_prefix[i] +", "
                        else:
                            st = st + long_prefix[i] +"] "

                    for n in range(1, data.cls + 1):
                        st = st + str(class_recurr[n])+" "

                    st = st + str(round(higher_recurr,5))
                    print(st)

        higher_recurr = next_recurr(infog_recurr, higher_recurr)


# # ############
# # # Inginious
# ############
# def main():
#     pos_filepath = sys.argv[1] # filepath to positive class file
#     neg_filepath = sys.argv[2] # filepath to negative class file
#     k = int(sys.argv[3])
#     # TODO: read the dataset files and call your miner to print the top k itemsets
#     PrefixSpan_Sup_Clos_Info_Gain(pos_filepath, neg_filepath, k)


# if __name__ == "__main__":
#     main()

#########################
# Initiate Algo Locally
#########################
neg_filepath = "/mnt/c/Users/b_tib/coding/Msc/oLING2364/Assignements/sequence-mining-pattern/Datasets/Protein/SRC1521.txt"
pos_filepath = "/mnt/c/Users/b_tib/coding/Msc/oLING2364/Assignements/sequence-mining-pattern/Datasets/Protein/PKA_group15.txt"

k = 10
PrefixSpan_Sup_Clos_Info_Gain(pos_filepath, neg_filepath, k)

