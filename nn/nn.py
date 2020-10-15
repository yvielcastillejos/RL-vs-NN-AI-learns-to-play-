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

#traindata = np.load("/Users/yvielcastillejos/gym/data.npy", allow_pickle=True)
#X = np.array([i[0] for i in traindata])
#y =np.expand_dims(np.array([np.argmax(i[1]) for i in traindata]), axis = 1)

#y_tens = torch.from_numpy(X).float()
#y_tens = torch.from_numpy(y).float()
#X_tensor = X_tens[0:17000]
#y_tensor = y_tens[0:17000]
#X_validation = X_tens[17000:len(X_tens)-1]
#y_validation = X_tens[17000:len(y_tens)-1]


#print(np.shape(y))
#print(np.shape(X))

class NN(nn.Module):
   def __init__(self):
       super(NN, self).__init__()
       self.fc1 = nn.Linear(2, 100)
       self.fc2 = nn.Linear(100,3)
   def forward(self, x):
       x = F.relu(self.fc1(x))
       x = F.log_softmax(self.fc2(x), dim = 1)
       return x

def initialize(learningrate):
   torch.manual_seed(1)
   model = NN()
   loss = torch.nn.CrossEntropyLoss()
   opt = torch.optim.SGD(model.parameters(), lr= learningrate)
   return model, loss, opt

def accuracy(predict, trainlabel):
  # print(np.shape(predict.detach().numpy()))
   predict = np.expand_dims(np.array([np.argmax(i) for i in predict.detach().numpy()]), axis =1)
  # print(predict)
#   print(np.clip(abs(predict-trainlabel.numpy()),0,1))
   a = 1 - np.sum(np.clip(abs(predict-trainlabel.numpy()),0,1))/len(trainlabel)
   return a

def train(traindata, trainlabel, validationdata, validationlabel):
   lossRec = []
   vlossRec = []
   nRec = []
   trainAveRec = []
   trainAccRec = []
   validAccRec = []
   validfreqRec = []
   trainsum = []
   mdl, lsf, op = initialize(0.1)
   for i in range(7):
       print(f"EPOCH {i}")
       for k in range(169):
       # Training Loop
          traindata1 = traindata[100*k:100*k+100]
          trainlabel1 = trainlabel[100*k:k*100+100]
          op.zero_grad()
          predict = mdl(traindata1.float())
          ls = lsf(input = predict.squeeze(), target = trainlabel1.float().squeeze().long())
          ls.backward()
          op.step()
          # Training Accuracy Calculation
          t_predict = mdl(traindata)
          trainAcc =  accuracy(t_predict, trainlabel)
          trainAccRec.append(trainAcc)
          # Validation Accuracy Calculation
          v_predict = mdl(validationdata.float())
          validAcc = accuracy(v_predict, validationlabel)
          validAccRec.append(validAcc)
          if k%100 == 0:
                print(f"Then training accuracy is {trainAcc:2f}, the validation accuracy is {validAcc: .2f} at batches # {k}")
   return mdl, trainAccRec, validAccRec

#if __name__ == "__main__":
traindata = np.load("/Users/yvielcastillejos/gym/data.npy", allow_pickle=True)
X = np.array([i[0] for i in traindata])
y =np.expand_dims(np.array([np.argmax(i[1]) for i in traindata]), axis = 1)

X_tens = torch.from_numpy(X).float()
y_tens = torch.from_numpy(y).float()
X_tensor = X_tens[0:17001]
y_tensor = y_tens[0:17001]
X_validation = X_tens[17000:len(X_tens)-1]
y_validation = y_tens[17000:len(y_tens)-1]
global model
global Tacc
global Vacc
model, Tacc, Vacc = train(X_tensor, y_tensor, X_validation, y_validation) 
