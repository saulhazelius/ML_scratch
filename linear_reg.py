import random
import numpy as np
import data
import matplotlib.pyplot as plt
# linear regression using Stochastic Gradient Descent
# polynomial degree = 3 h(thet,x) = t0 + t1*x + t2*x**2 + ...
thet0 = random.uniform(-0.5,0.5)
thet1 = random.uniform(-0.5,0.5)
thet2 = random.uniform(-0.5,0.5)
thet3 = random.uniform(-0.5,0.5)
#thet4 = random.uniform(-1,1)

X,Y = data.data_generation()
lr = 0.3
J=[] # cost funtion to minimize
m = len(X) # samples
cc = 0.001 # convergence criterion
# initial  j
sumd = 0
for i in range(m):


	h = thet0 + thet1*X[i] + thet2*X[i]**2 + thet3*X[i]**3
	dif = (h - Y[i])**2
	sumd = sumd + dif
j = (1/(2*m))*sumd
ite = []
for itera in range(10000):
	pastj = j
	past0 = thet0
	past1 = thet1
	past2 = thet2
	past3 = thet3
	sumd = 0
	for i in range(m):
		dhx0 = thet0+thet1*X[i]+thet2*X[i]**2+thet3*X[i]**3-Y[i]
		dhx1 = (thet0 +thet1*X[i]+thet2*X[i]**2+thet3*X[i]**3-Y[i])*X[i]
		dhx2 = (thet0 +thet1*X[i]+thet2*X[i]**2+thet3*X[i]**3-Y[i])*X[i]**2
		dhx3 = (thet0 +thet1*X[i]+thet2*X[i]**2+thet3*X[i]**3-Y[i])*X[i]**3
		temp0 = thet0 - lr*dhx0
		temp1 = thet1 - lr*dhx1
		temp2 = thet2 - lr*dhx2
		temp3 = thet3 - lr*dhx3

		thet0 = temp0
		thet1 = temp1
		thet2 = temp2
		thet3 = temp3
		h = thet0 + thet1*X[i] + thet2*X[i]**2 + thet3*X[i]**3
		dif = (h - Y[i])**2
		sumd = sumd+dif
	j = (1/(2*m))*sumd	
	J.append(j)
	ite.append(itera)
	
	if thet0 - past0 < cc and thet1 - past1 < cc and  thet2 - past2 < cc and  thet3 - past3 < cc:	
		break
Yp = [] # output predicted
sin = []
Xs = sorted(X)
for ii in range(len(Xs)):
        Yp.append(thet0+thet1*Xs[ii]+thet2*Xs[ii]**2+thet3*Xs[ii]**3)
        sin.append(np.sin(2*np.pi*Xs[ii]))
Yp = np.array(Yp)
sin = np.array(sin)
print("Final parematers",thet0,thet1,thet2,thet3)
plt.plot(Xs,sin,color='green',label='Sin2pi*x')
plt.scatter(X,Y,label='samples')
plt.plot(Xs,Yp,color='red',label='Y predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Degree = 3 alpha= '+str(lr))
plt.legend()
plt.savefig('opt_d3_a_'+str(lr)+'.pdf',format='pdf')
fig, ax = plt.subplots()
ax.plot(ite,J,label=str(itera)+' iterations degree=3 alpha= '+str(lr))
plt.xlabel('iterations')
plt.ylabel('error')
ax.legend(loc='upper center')
plt.savefig('error_d3.pdf',format='pdf')
print("final error= "+str(j))
