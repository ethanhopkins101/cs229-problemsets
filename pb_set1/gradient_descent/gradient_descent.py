import numpy as np
import matplotlib.pyplot as plt
import math

class GradientDescent():
    def __init__(self):
        self.x_train=self.y_train=self=self.theta=0

    def gradient_descent(self,x_train,y_train,lr,iteration):
        self.theta=np.array([[0 for _ in range(x_train.shape[1])]]).T
        lr=0.001
        iteration
        for i in range(iteration):
            hyp=np.dot(self.x_train,self.theta)
            dj=np.dot((hyp-self.y_train).T,self.x_train).T
            self.theta=self.theta-np.multiply(lr,dj)


    def fit(self,train,lr,iteration):
        self.x_train=train[:,:-1]
        self.y_train=train[:,-1]
        self.y_train=np.reshape(self.y_train,(self.y_train.shape[0],1))
        self.gradient_descent(self.x_train,self.y_train,lr,iteration)



    def predict(self,predict):
        x_pred=predict[:,:-1]
        y_pred=predict[:,-1]
        y_pred=np.reshape(y_pred,(y_pred.shape[0],1))
        if self.theta.shape[0]==x_pred.shape[0]:
          hyp=np.dot(x_pred,self.theta)
          cost_func=(1/2)*np.sum(np.square(hyp-y_pred))
          if self.theta.shape[1]==2:
                plt.figure(figsize=(9,6))
                plt.scatter(x_pred,y_pred)
                plt.plot(x_pred,self.theta[0]+self.theta[1]*x_pred)
                plt.show()
        