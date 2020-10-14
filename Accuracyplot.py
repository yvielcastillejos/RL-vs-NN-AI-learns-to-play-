from nn import Vacc, Tacc
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
def plot():
	validAccRec =Vacc
	trainAccRec = Tacc
	figure = plt.figure()
	x1 = np.arange(0,len(validAccRec),1)
	trainAccRec = savgol_filter(Tacc, 51,3)
	x1 = np.arange(0,len(trainAccRec),1)
	trainAccRec = savgol_filter(trainAccRec, 51,3)
	plt.plot(x1, trainAccRec)
#	holdon
	x5 = np.arange(0,len(validAccRec),1)
	y_2 = savgol_filter(validAccRec, 51,5)
	plt.plot(x5, y_2)
	plt.xlabel(f"step ({100} samples per step)")
	plt.ylabel("Accuracy on one batch at a time")
	plt.title(f"Accuracy vs Mini-Batches (with Learning rate: {0.1} and batch size:{ 0.1} epoch: {7})")
	plt.legend(["Training Accuracy", "Validation Accuracy"])
	plt.show()

if __name__ == "__main__":
	plot()
