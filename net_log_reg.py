import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import data_pre
import matplotlib.pyplot as plt

# same problem of simple_log_reg using PyTorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X1, X2, Y, Xp, Xn, Yp, Yn = data_pre.data_generation()

lr = 0.1

X_train = np.array([X1,X2])

X_train = torch.tensor(X_train.transpose())

class Net(nn.Module):
        def __init__(self):
                super(Net,self).__init__()

                self.f1 = nn.Linear(2,1) # 2 features  
                self.l1 = nn.Sigmoid() # classification 0,  1
        def forward(self,x):
                x = self.f1(x)
                x = self.l1(x)
                return x

classif = Net().to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(classif.parameters(), lr=lr )
epochs =240
ite =[]
l =[]

for epo in range(epochs):
	for b in range(len(X_train)):
		X = torch.tensor(X_train[b],dtype=torch.float)
		optimizer.zero_grad()
		prediction = classif(X)
		label = torch.tensor(Y[b],dtype=torch.float).unsqueeze(0)
		loss = criterion(prediction,label)
		loss.backward()
		optimizer.step()
	ite.append(epo)
	l.append(loss.item())
	print(epo,loss.item())



par=[]
for param in classif.parameters():
	par.append(np.array(param.data))

p = np.array(par[0]).reshape(2,) # p[0] parameters p[1] bias
bias = np.array(par[1])
ypred = []
for i in range(len(X_train)):
	ypred.append((-1/p[1])*(bias+p[0]*X1[i]))
ypred=np.array(ypred)	
plt.scatter(Xp,Yp,label='1')
plt.scatter(Xn,Yn,label='0')
plt.plot(X1,ypred,color='black',label='Y predicted')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

# plot error and save fig
fig, ax = plt.subplots()
ax.plot(ite,l,label=str(epochs)+' iterations alpha= '+str(lr))
plt.xlabel('iterations')
plt.ylabel('error')
ax.legend(loc='upper center')
plt.savefig('error.pdf',format='pdf')

