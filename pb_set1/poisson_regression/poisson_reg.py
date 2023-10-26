import numpy as np

class Poisson():
    def __init__(self):
        self.x_train=self.y_train=self=self.theta=0
        self.poisson_reg=0

    def fit(self,train,lr,iteration):
        self.x_train=train[:,:-1]
        self.theta=np.array([[0 for _ in range(self.x_train.shape[1])]]).T
        self.y_train=train[:,-1]
        self.y_train=np.reshape(self.y_train,(self.y_train.shape[0],1))
        for i in range(iteration):
            hyp=np.exp(self.x_train,self.theta)
            theta=theta-lr*(np.dot(self.y_train-hyp,self.x_train))


    def predict(self,predict):
        x_pred=predict[:,:-1]
        y_pred=predict[:,-1]
        y_pred=np.reshape(y_pred,(y_pred.shape[0],1))
        predictions=[]
        for i in range(x_pred.shape[0]):
            probability=(1/y_pred[i])*(np.exp(np.dot(np.dot(x_pred[i,:],self.theta),y_pred[i]))-np.exp(np.dot(x_pred[i,:],self.theta)))
            predictions.append(probability)
        return predictions