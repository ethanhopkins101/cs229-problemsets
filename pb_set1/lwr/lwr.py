import numpy as np
import matplotlib.pyplot as plt
import math

class Lwr():
    def __init__(self):
        self.x_train=self.y_train=self.x_pred=self.y_pred= self.x_valid=self.y_valid=0
        self.theta=self.weights=0

    def fit(self,t,train,predict):
        self.t=t
        self.x_train=train[:,:-1]
        self.y_train=train[:,-1]
        self.y_train=np.reshape(self.y_train,(self.y_train.shape[0],1))
        self.x_pred=predict[:,:-1]
        self.y_pred=predict[:,-1]
        self.y_pred=np.reshape(self.y_pred,(self.y_pred.shape[0],1))
        self.list_theta=[]
        for i in range(self.x_pred.shape[0]):
            distances=np.linalg.norm(self.x_train-self.x_pred[i,:])
            self.weights=np.exp(-distances**2/(2*t**2))
            self.weights=np.diag(self.weights)
            fst=np.dot(np.dot(self.x_train.T,self.weights),self.x_train)
            snd=np.dot(np.dot(self.x_train.T,self.weights),self.y_train)
            self.theta=np.dot(np.linalg.inv(fst),snd)
            self.list_theta.append(self.theta)
        return self.list_theta


    def predict(self,valid):
        self.x_valid=valid[:,:-1]
        self.y_valid=valid[:,-1]
        self.y_valid=np.reshape(self.y_valid,(self.y_valid.shape[0],1))
        list_predictions=[]
        for i in range(self.x_valid.shape[0]):
            list_predictions.append(np.dot(self.list_theta[i].T,self.x_valid[i,:].T))
        plt.figure(figsize=(9,6))
        plt.scatter(self.x_valid,list_predictions,color='lightgreen')
        plt.scatter(self.x_valid,self.y_valid,color='green')
        plt.title('Predicted Values relative to True values',weight='bold')
