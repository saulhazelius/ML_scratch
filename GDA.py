import os
import glob
import random
import numpy as np
from skimage import io
dp = glob.glob(os.getcwd()+"/positives/*png")
dn = glob.glob(os.getcwd()+"/negatives/*png")

# GDA Gaussian Discriminant Analysis for image classification. The positive and negative png samples are in the positives.zip and negatives.zip files respectively.


def split(X,v): 
        """generates partitions for the cross validation s-fold (s = v)"""
        pair=[]
        for k in range(len(X)):
		
                pair.append(list((list(X[k]))))
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

fp = []

# read images 
for sam in dp: # positive images 
	im = io.imread(sam) # ndarray uint8
	im = np.array(im,dtype='int16')	
	fp.append(im)

fp = np.array(fp)

fn = []
for sam in dn: # negative images  
	im = io.imread(sam)
	im = np.array(im,dtype='int16')	
	fn.append(im)

fn = np.array(fn)

hlistp,pairp= split(fp,v) # sfold huge list of 10 lists each of size 3 ( 30 pos samples / 10-fold = 3)
hlistn,pairn= split(fn,v) # sfold huge list




for i in range(len(hlistn)):
 	# Here comes the features. Features only based on colors. Mean of R, G, B values and mean of mins and maxs
	X1 =[] # meanR
	X2 =[] # meanG
	X3 =[] # meanB
	X4 =[]# mean (minR minG minB)
	X5 =[] # mean (maxR maxG maxB)
	#The same for negative examples
	X1n =[] # meanR
	X2n =[]
	X3n =[]
	X4n =[]# mean (minR minG minB)
	X5n =[] # mean (maxR maxG maxB)
	l = list(range(10))
	l.remove(i)
	valp = hlistp[i]
	trainp = [hlistp[j] for j in l]
	valn = hlistn[i]	
	trainn = [hlistn[j] for j in l]
	l.insert(i,i)	
	train = trainp + trainn
	val = valp + valn
	train = np.array(train)
	val = np.array(val)
	v = len(val)
	tp = 0 # true positives, true negatives, false posites, and false negatives for the accuracy and confusion matrix
	tn = 0
	fp = 0
	fn = 0
	for k in range(len(train)):
		
		Y = train[k]
		for e in range(len(Y)):
			X=Y[e]
			if k < 9: 
				X1.append(np.mean(X[:,:,0]))
				X2.append(np.mean(X[:,:,1]))
				X3.append(np.mean(X[:,:,2]))
				ap = np.mean( np.min(X[:,:,0]) + np.min(X[:,:,1]) + np.min(X[:,:,2]) )
				X4.append(ap)
				ap2 = np.mean( np.max(X[:,:,0]) + np.max(X[:,:,1]) + np.max(X[:,:,2]) )
				X5.append(ap2)
				
			else:
				X1n.append(np.mean(X[:,:,0]))
				X2n.append(np.mean(X[:,:,1]))
				X3n.append(np.mean(X[:,:,2]))
				ap = np.mean( np.min(X[:,:,0]) + np.min(X[:,:,1]) + np.min(X[:,:,2]) )
				X4n.append(ap)
				ap2 = np.mean( np.max(X[:,:,0]) + np.max(X[:,:,1]) + np.max(X[:,:,2]) )
				X5n.append(ap2)
			

	phi = 0.5 # phi1 = phi2 30 pos and 30 neg
	mu0 = np.array([np.mean(X1n),np.mean(X2n),np.mean(X3n),np.mean(X4n),np.mean(X5n)])#mean array neg
	mu1 = np.array([np.mean(X1),np.mean(X2),np.mean(X3),np.mean(X4),np.mean(X5)])# mean array pos 

	summ=np.zeros(shape=(5,5))
	m =27 
	for ii in range(m):
		a = np.array([X1[ii], X2[ii],X3[ii],X4[ii],X5[ii]])
		b = np.array([X1n[ii], X2n[ii],X3n[ii],X4n[ii],X5n[ii]])
		dif1 = (b - mu0).reshape(-1,1)
		dif1t = np.transpose(dif1)
		mm = np.matmul(dif1,dif1t)
		dif2 = (a - mu1).reshape(-1,1)
		dif2t = np.transpose(dif2)
		mm2 = np.matmul(dif2,dif2t)
		summ = summ + mm + mm2
	

	sig = summ/m*2
	print("partition= ",str(i+1))
	print('mu0= ',mu0,'mu1= ',mu1)
	print('sigma= ',sig)
	siginv=np.linalg.inv(sig)
	det = np.linalg.det(sig)
	### another loop to estimate p(x|y) for validation set
	X1v = []
	X2v = []
	X3v = []
	X4v = []
	X5v = []
	X1vn = []
	X2vn = []
	X3vn = []
	X4vn = []
	X5vn = []
	for k in range(v):
                
		print('Sample ',str(k+1))
		if k < 3:
			X1v.append(np.mean(val[k][:,:,0]))
			X2v.append(np.mean(val[k][:,:,1]))
			X3v.append(np.mean(val[k][:,:,2]))
			X4v.append(np.min(val[k][:,:,0]) + np.min(val[k][:,:,1]) + np.min(val[k][:,:,2]))
			X5v.append(np.max(val[k][:,:,0]) + np.max(val[k][:,:,1]) + np.max(val[k][:,:,2]))
			a = np.array([X1v[k], X2v[k], X3v[k], X4v[k], X5v[k]])
		else:
			X1vn.append(np.mean(val[k][:,:,0]))
			X2vn.append(np.mean(val[k][:,:,1]))
			X3vn.append(np.mean(val[k][:,:,2]))
			X4vn.append(np.min(val[k][:,:,0]) + np.min(val[k][:,:,1]) + np.min(val[k][:,:,2]))
			X5vn.append(np.max(val[k][:,:,0]) + np.max(val[k][:,:,1]) + np.max(val[k][:,:,2]))
			a = np.array([X1vn[k-3], X2vn[k-3], X3vn[k-3], X4vn[k-3], X5vn[k-3]])
		dif1 = (a - mu0).reshape(-1,1)
		dif2 = (a - mu1).reshape(-1,1)
		dif1t = np.transpose(dif1)
		dif2t = np.transpose(dif2)
		p1 = np.matmul(siginv,dif1)
		pp1 = np.dot(dif1t,p1)
		p2 = np.matmul(siginv,dif2)
		pp2 = np.dot(dif1t,p2)
		prob1 = (1/((2*np.pi)**(m/2)*det**(0.5)))*np.exp(-0.5*pp1) # p(x|y=0)
		prob2 = (1/((2*np.pi)**(m/2)*det**(0.5)))*np.exp(-0.5*pp2) # p(x|y=1)

		
		if k < 3:
			print('real=positive')
			
			print('y=0 score = ',prob1*0.5) # we need argmax (p(x|y=0) * p(y)) orginal Bayes: (p(x|y=0) * p(y))/p(x) = p(y=0|x)
			print('y=1 score = ',prob2*0.5) # we need argmax (p(x|y=1) * p(y)) orginal Bayes: (p(x|y=1) * p(y))/p(x) = p(y=1|x)
			if prob2>prob1:
				print('predicted=positive')
				tp += 1
			else:
				print('predicted=negative')
				fn += 1
				
		
		else:
			print('real=negative')

			print('y=0 score = ',prob1*0.5) # we need argmax (p(x|y=0) * p(y)) orginal Bayes: (p(x|y=0) * p(y))/p(x) = p(y=0|x)
			print('y=1 score = ',prob2*0.5) # we need argmax (p(x|y=1) * p(y)) orginal Bayes: (p(x|y=1) * p(y))/p(x) = p(y=1|x)
			if prob1>prob2:
				print('predicted=negative')
				tn += 1
			else:
				print('predicted=positive')
				fp += 1
	print('confusion matrix: ')
	print(str(np.array([[tn,fp],[fn,tp]])))
	print('accuracy= ',str(round((tp+tn)/(tp+fp+tn+fn),2)))
	


