"""
Skeleton file for the project 1 of the LINGI2364 course.
Use this as your submission file. Every piece of code that is used in your program should be put inside this file.

This file given to you as a skeleton for your implementation of the Apriori and Depth
First Search algorithms. You are not obligated to use them and are free to write any class or method as long as the
following requirements are respected:

Your apriori and alternativeMiner methods must take as parameters a string corresponding to the path to a valid
dataset file and a double corresponding to the minimum frequency.
You must write on the standard output (use the print() method) all the itemsets that are frequent in the dataset file
according to the minimum frequency given. Each itemset has to be printed on one line following the format:
[<item 1>, <item 2>, ... <item k>] (<frequency>).
Tip: you can use Arrays.toString(int[] a) to print an itemset.

The items in an itemset must be printed in lexicographical order. However, the itemsets themselves can be printed in
any order.

Do not change the signature of the apriori and alternative_miner methods as they will be called by the test script.

__authors__ = "<Breno Tiburcio>"
"""
import time
import copy
import matplotlib.pyplot as plt

class Dataset:
    """Utility class to manage a dataset stored in a external file."""
    def __init__(self, filepath):
        """reads the dataset file and initializes files"""
        self.transactions = list()
        self.items = set()

        try:
            lines = [line.strip() for line in open(filepath, "r")]
            lines = [line for line in lines if line]  # Skipping blank lines
            for line in lines:
                transaction = list(map(int, line.split(" ")))
                self.transactions.append(transaction)
                for item in transaction:
                    self.items.add(item)
        except IOError as e:
            print("Unable to read dataset file!\n" + e)

    def trans_num(self):
        """Returns the number of transactions in the dataset"""
        return len(self.transactions)

    def items_num(self):
        """Returns the number of different items in the dataset"""
        return len(self.items)

    def get_transaction(self, i):
        """Returns the transaction at index i as an int array"""
        return self.transactions[i]
    def get_itemsets(self,idx):
        return list(self.items)[idx]
    def get_Allitemsets(self):
        return list(self.items)




##################################
# APRIORI
##################################

def get_candidates(itemsets):
    # print(itemsets)
    # startime = time.perf_counter()
    cand=[]
    startime = time.perf_counter()
    for i in range(len(itemsets)-1):
        for j in range(i + 1, len(itemsets)):
            if len(itemsets[i]) == 1:
                cand.append([itemsets[i][0]])
                cand[-1].append(itemsets[j][0])
            elif itemsets[i][:-1] == itemsets[j][:-1]:
                cand.append(itemsets[i][:-1]) # cand([12])
                cand[-1].append(itemsets[i][-1]) # cand([12,15])
                cand[-1].append(itemsets[j][-1]) # cand([12,15,16])
    # endtime = time.perf_counter()
    # print('Elapsed Time:', endtime-startime)
    return cand

def get_freq(itemset, transactions, n_trans):
	count = 0
	tempItem=set(itemset)
	for t in range(n_trans): # At each transaction "t" in transactions
		if tempItem.issubset(transactions[t]): # it takes O(n)
			count += 1
	return count/n_trans

def apriori(filepath, minFrequency):
    """Runs the apriori algorithm on the specified file with the given minimum frequency"""
    #data = Dataset(filepath) # for individual dataset run
    data = filepath

    # itemset can be formed by merging number. New number would be created and teh line.
    items = [[x] for x in data.items] # level[[items]]
    transactions = data.transactions  # list of transaction
    n_trans = data.trans_num()

    #print(len(items), n_trans)
    level_key=0
    frequents = {} # {level:[[items]]}
    while len(items) > 0:
        level_val = []
        # Test each item frequency
        for i in range(len(items)):
            # test item[i] frequency vel3:[itemset[items,freq]]}
            item_freq = get_freq(items[i], transactions, n_trans)
            # if above support
            if item_freq >= minFrequency :
                #print(items[i],'('+str(item_freq)+')')
                level_val.append([items[i],item_freq])
        frequents[level_key] = level_val
        if len(level_val) > 1 : # remove < 4
            level_key += 1
            old_items = [x[0] for x in level_val]
            if len(old_items) > 1:
                items = get_candidates(old_items)
        else:
            break
    return


##################################
# APRIORI with Vertical Rep.
###################################

def vertical_intersect(Vertical_rep_frequent,result_list,n_trans):
    # it optimizes the algorithm by more than 4x
    fixed=result_list[0]
    intersect=Vertical_rep_frequent.get(frozenset({fixed}),set())
    for i in result_list:
        intersect=intersect.intersection(Vertical_rep_frequent.get(frozenset({i}),set()))
        if len(intersect)==0:
            break
    return len(intersect)/n_trans

def get_vp(dataset):
    verticat_rep={}
    for idx in range(dataset.trans_num()):
        transaction=dataset.get_transaction(idx)
        for item in transaction:
            if frozenset([item]) not in verticat_rep:
                verticat_rep[frozenset([item]) ]=set([])
            verticat_rep[frozenset([item]) ].add(idx)
    return verticat_rep

def vert_apriori(filepath, minFrequency):
    """Runs the apriori algorithm on the specified file with the given minimum frequency"""
    #data = Dataset(filepath) # for individual dataset run
    data = filepath # to speed up algo comparisson

    vp = get_vp(data) # returns vertical rep. dictionairy
    # itemset can be formed by merging number. New number would be created and teh line.
    items = [[x] for x in data.items] # level[[items]]
    transactions = data.transactions  # list of transaction
    n_trans = data.trans_num()

    level_key=0
    frequents = {} # {level:[[items]]}
    while len(items) > 0:
        level_val = []
        # Test each item frequency
        for i in range(len(items)):
            item_freq=vertical_intersect(vp,items[i],n_trans)
            # if above support
            if item_freq >= minFrequency :
                #print(items[i],'('+str(item_freq)+')')
                level_val.append([items[i],item_freq])
        frequents[level_key] = level_val
        if len(level_val) > 1 : # remove < 4
            level_key =+1
            old_items = [x[0] for x in level_val]
            if len(old_items) > 1:
                items = get_candidates(old_items)
        else:
            break
    return

##################################
# ECLAT
##################################

def get_vp(dataset): # compute vertical representation
    verticat_rep={}
    for idx in range(dataset.trans_num()):
        transaction=dataset.get_transaction(idx)
        for item in transaction:
            if frozenset([item]) not in verticat_rep:
                verticat_rep[frozenset([item]) ]=set([])
            verticat_rep[frozenset([item]) ].add(idx)
    return verticat_rep

def vr_intersect(vr,result_list):
    fixed=result_list[0]
    intersect=vr.get(frozenset({fixed}),set())
    for i in result_list:
        intersect=intersect.intersection(vr.get(frozenset({i}),set()))
        if len(intersect)==0:
            break
    #return len(intersect)/n_trans
    return intersect

def depthFirstSearch(item,data,minFrequency, total_trans,vp,Itemsets,explored):
    exploredLocal=explored.copy()
    if type(item)!=list:
        item=[item]
        #print(item)
    while (len(exploredLocal)>0):
        item2=exploredLocal.pop()
        #print(iem2]

        if item2 not in item:
            item2=[item2]
            item2=item+item2
            intersect=vr_intersect(vp,item2)
            supp=len(intersect)/total_trans

            if(supp >= minFrequency):
                itemsorted=sorted(item2)
                if tuple(itemsorted) not in Itemsets:
                    Itemsets[tuple(itemsorted)]=supp
                    #print(sorted(item2), '('+str(supp)+')')
                    depthFirstSearch(item2,data,minFrequency, total_trans,vp,Itemsets,exploredLocal)

    return None

def alternative_miner(filepath, minFrequency):
    """Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""

    #data = Dataset(filepath) # for individual dataset run
    data = filepath # to speed up algo comparisson
    vp=get_vp(data)

    #print(vp)
    total_trans = data.trans_num()
    Itemsets={}
    explored=data.get_Allitemsets()
    for idx in range(data.items_num()):
        item=data.get_itemsets(idx)

        ExploredNew=explored.copy()
        ExploredNew.pop()
        supp=(len(vp[frozenset([item])])/data.trans_num())
        if supp>=minFrequency:
            #print(sorted([item]), '('+str(supp)+')')
            depthFirstSearch(item, data, minFrequency, total_trans,vp,Itemsets,ExploredNew)
    return None


##################################
# Running
##################################

count=0

path = '/mnt/c/Users/b_tib/coding/Msc/oLING2364/Assignements/itemset-mining-pattern/Datasets/'
datasets=['toy', 'chess'] # , 'chess','mushroom', 'pumsb_star', 'pumsb','connect', 'retail' ,'accidents'
minFrequencies = [0.95,0.9,0.85,0.8] # 0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.02,0.01
algos = [apriori, vert_apriori, alternative_miner]

elapses_aprior = {}
elapses_vert_aprior = {}
elapses_eclat = {}

for db in datasets:
    data = Dataset(str(path) + str(db) + '.dat')    # rememenber to change data = filepath !
    #print(data.transactions)
    elapses_aprior[db] = []
    elapses_vert_aprior[db] = []
    elapses_eclat[db] = []

    for minFrequency in minFrequencies:

        startime = time.perf_counter()
        apriori(data, minFrequency)
        endtime = time.perf_counter()
        #print('elapses_aprior: ', str(minFrequency) + ' - ', str(time-startime))
        elapses_vert_aprior[db].append(endtime-startime)

        startime = time.perf_counter()
        vert_apriori(data, minFrequency)
        endtime = time.perf_counter()
        #print('elapses_vert_aprior: ', str(minFrequency)+ ' - ', str(time-startime))
        elapses_vert_aprior[db].append(endtime-startime)

        startime = time.perf_counter()
        alternative_miner(data, minFrequency)
        endtime = time.perf_counter()
        #print('elapses_eclat: ', str(minFrequency)+ ' - ', str(time-startime))
        elapses_vert_aprior[db].append(endtime-startime)

###########
# Printing
###########
for db in datasets:
    apri = elapses_aprior[db].copy()
    vapri = elapses_vert_aprior[db].copy()
    eclat = elapses_eclat[db].copy()

    plt.figure(count)
    #fig.suptitle('Run-time comparison')
    plt.plot(rng[:len(apri)],ap,'-+',label='Apriori.')
    plt.plot(rng[:len(vapri)],gr,'-^',label='Vert. Apriori.')
    plt.plot(rng[:len(eclat)],uap,'->',label='Eclat')
    plt.title('Dataset: '+db)
    plt.xlabel('Minimum Threshold')
    plt.ylabel('Run-time in Sec.')
    plt.legend()
    plt.savefig(db+'-plot.png')
    count+=1

