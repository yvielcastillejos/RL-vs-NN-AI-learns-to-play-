import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import mountaincar
#import sklearn as sk
#from sklearn import preprocessing, model_selection
import torch.nn.functional as F

traindata = np.load("/Users/yvielcastillejos/gym/data.npy", allow_pickle=True)
X = np.array([i[0] for i in traindata])
y = np.array([i[1] for i in traindata])
print(y[506:520])
class NN(nn.Module):
   def __init__(self):
       super(NN, self).__init__()
       self.fc1 = nn.Linear(2, 258)
       self.fc2 = nn.Linear(258,3)
   def forward(self, x):
       x = F.relu(self.fc1(x))
       x = F.softmax(self.fc2(x))
       return x

def initialize(learningrate):
   torch.manual_seed(1)
   model = NN()
   loss = torch.nn.CrossEntropyLoss()
   opt = torch.optim.SGD(imodel.parameters(), lr= learningrate)
   return model, loss, opt

def main():
   lossRec = []
   vlossRec = []
   nRec = []
   trainAveRec = []
   trainAccRec = []
   validAccRec = []
   validfreqRec = []
   trainsum = []
   mdl, ls, op = initialize(0.1)
   for i in range(100):
       op.zero_grad()
       predict = mdl(traindata)
       ls = ls(input = predict.squeeze(), target = trainlabel.float().squeeze())
       ls.backward()
       op.step()
       trainAcc = accuracy(predict, trainlabel)
       trainAccRec.append(trainAcc)
       if i%10 == 0:
            print(f"The training accuracy is {trainAcc} at epoch {i}")
   return mdl, trainAccRec

if __name__ == "__main__":
   main() 
