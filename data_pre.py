import matplotlib.pyplot as plt

def data_generation():
	f = open('data.txt','r')

	X1 = []
	X2 = []
	Xp = []
	Xn = []
	Yp = []
	Yn = []
	Y = []
	for line in f:
		X1.append(float(line.split()[0]))
		if int(line.split()[2]) == 0:
			Xn.append(float(line.split()[0]))
			Yn.append(float(line.split()[1]))
		if int(line.split()[2]) == 1:
			Xp.append(float(line.split()[0]))
			Yp.append(float(line.split()[1]))
			
		X2.append(float(line.split()[1]))
		Y.append(int(line.split()[2]))
	return X1, X2, Y, Xp, Xn, Yp, Yn
#X1, X2, Y, Xp, Xn, Yp, Yn = data_generation()

#plt.scatter(Xp,Yp)
#plt.show()
