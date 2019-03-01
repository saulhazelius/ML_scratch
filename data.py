import random
import numpy as np
from math import sin, pi
import matplotlib.pyplot as plt

def data_generation():
	"""function to give data from y = sin2pix + eps"""
	Y = []
	X = []
	for i in range(100):
		x = random.random() # random from [0,1)
		X.append(x)
		eps = random.uniform(-0.3,0.3)
		y = sin(2*pi*x) + eps
		Y.append(y)
	X = np.array(X)
	Y = np.array(Y)

	return X, Y 


def plot_data():
	
	X, Y = data_generation()
	plt.scatter(X,Y)
	plt.xlabel('x')
	plt.ylabel('y = sin(2pix) + eps')
	plt.show()

