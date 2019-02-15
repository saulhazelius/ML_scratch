import random
import numpy as np
import data_pre
import matplotlib.pyplot as plt
# model : z(theta,x) = theta0 + theta1*x + theta2*x .
# h = 1/(1+exp(-z))
thet0 = random.uniform(-0.01,0.01)
thet1 = random.uniform(-0.01,0.01)
thet2 = random.uniform(-0.01,0.01)

X1, X2, Y, Xp, Xn, Yp, Yn = data_pre.data_generation()



lr = 0.1
J=[] # cost funtion to minimize
m = len(X1) # samples
cc = 0.01
# initial j
sumd = 0
Yini = [] # to initial plot
for i in range(m):
	Yini.append((-1/thet2)*(thet0+thet1*X1[i]))

	z = thet0 + thet1*X1[i] + thet2*X2[i]
	h = 1/(1+np.exp(-z))
	dif = (h - Y[i])**2
	sumd = sumd + dif
j = (1/(2*m))*sumd
ite = []
for itera in range(10000):
	pastj = j
	past0 = thet0
	past1 = thet1
	past2 = thet2
	sumd = 0
	for i in range(m):
		z = thet0 + thet1*X1[i] + thet2*X2[i]
		h = 1/(1+np.exp(-z))
		dhx0 = h-Y[i]
		dhx1 = (h-Y[i])*X1[i]
		dhx2 = (h-Y[i])*X2[i]
		temp0 = thet0 - lr*dhx0
		temp1 = thet1 - lr*dhx1
		temp2 = thet2 - lr*dhx2

		thet0 = temp0
		thet1 = temp1
		thet2 = temp2
		z = thet0 + thet1*X1[i] + thet2*X2[i]
		h = 1/(1+np.exp(-z))
		dif = (h - Y[i])**2
		sumd = sumd+dif
	j = (1/(2*m))*sumd	
	J.append(j)
	ite.append(itera)
	print(thet0,thet1,thet2)
	
	if thet0 - past0 < cc and thet1 - past1 < cc and  thet2 - past2 < cc:	
		break
Ypr = [] # output predicted X2
for ii in range(len(X1)):
	Ypr.append((-1/thet2)*(thet0+thet1*X1[ii]))
Ypr = np.array(Ypr)


plt.scatter(Xp,Yp,label='1')
plt.scatter(Xn,Yn,label='0')
plt.plot(X1,Ypr,color='black',label='Y predicted')
plt.plot(X1,Yini,color='blue',label='Y initial')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.savefig('opt_a_'+str(lr)+'.pdf',format='pdf')
fig, ax = plt.subplots()
ax.plot(ite,J,label=str(itera)+' iterations alpha= '+str(lr))
plt.xlabel('iterations')
plt.ylabel('error')
ax.legend(loc='upper center')
plt.savefig('error.pdf',format='pdf')
print("final error= "+str(j))
