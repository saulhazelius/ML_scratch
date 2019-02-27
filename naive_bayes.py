# naive bayes classifier on spam dataset from UCI ML repository 
# the data file freq_features.csv was created after removing the last 9 features and preserving only 48 word frequencies features. the data wre MinMax Scaled and lines without frequencies were removed.

import random
f = open('freq_features.csv','r')

def split(f,v):
        """generates list of pairs list"""
        pair=[]
        for line in f:
                pair.append(line)
        random.seed(32)
        random.shuffle(pair)
        splen = round(len(pair)/v)
        tt =[]
        for k in range(v):
                uu =[]
                for i in range(k*splen,(k+1)*splen):
                        if i == len(pair):
                                break
                        uu.append(pair[i])
                tt.append(uu)


        return tt, pair

v=10 # sfold

hlist,pair= split(f,v) # sfold huge list of pairs

spprob = float(1798/4437) # 1798 spam lines and 4437 total lines in file
nspprob = float(1) - spprob
def evaluate(val):
	c = 0
	tp = 0	
	tn = 0
	fp = 0
	fn = 0
	for h in range(len(val)):
		prsp = 1
		prnsp = 1
		for y in range(1,49):
			if float(val[h].split(',')[y-1]) > 0.0:
				prsp = prsp*ds[y]*spprob # here is the main core of naive bayes. prsp is a score obtained after the implementation of bayes theorem on independent distribution of words (p(Y|X) = p(X|Y)*p(Y)) where p(X|Y) assumes conditional independence: p(X|Y) = p(xi|Y)*p(xj|Y)*... 
				prnsp = prnsp*dns[y]*nspprob
		if prsp > prnsp:
			if val[h].split(',')[-1].strip() == str(1.0):
				c = c +1
				tp += 1
			else:
				fp += 1
		else:

			if val[h].split(',')[-1].strip() == str(0.0):
				
				c = c +1
				tn += 1
			else:
				fn += 1

	print("fold")
	print(str(k+1))
	print('percent correct')
	print(str(round(float(c/len(val)),2)))
	print("true pos", tp, "false pos", fp, "true neg", tn, "false neg",fn)



## cross validation 10 fold
for k in range(len(hlist)): # len hlist = 10
	ds = dict()
	dns = dict()
	for w in range(1,49): # dictionaries with 48 word per category
		ds[w] = 0
		dns[w] = 0
	val = hlist[k]
	train = [x for x in pair if x not in val] ## len(train) < 4437 - 444. it means there are no validation samples in the train set
	for t in range(len(train)):
		if train[t].split(',')[-1].strip() == str(1.0): # spam
			for j in range(1,49):
				ds[j] += float(train[t].split(',')[j-1])

		if train[t].split(',')[-1].strip() == str(0.0): # no spam
			for jj in range(1,49):
				dns[jj] += float(train[t].split(',')[jj-1])

	evaluate(val)

