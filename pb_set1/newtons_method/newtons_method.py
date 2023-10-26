import numpy as np
class Newtons():
    def __init__(self):
        self.x_train=self.y_train=self=self.theta=0
    
    def fit(self,train,iteration):
        self.x_train=train[:,:-1]
        self.theta=np.array([[0 for _ in range(self.x_train.shape[1])]]).T
        self.y_train=train[:,-1]
        self.y_train=np.reshape(self.y_train,(self.y_train.shape[0],1))
        m=self.x_train.shape[0]
        for i in range(iteration):
            hyp=1/(1+np.exp(self.x_train,self.theta))
            dj=(1/m)*(np.dot((hyp-self.y_train).T,self.x_train)).T
            hessian=(1/m)*(np.dot(np.dot(hyp.T,1-hyp),np.dot(self.x_train.T,self.x_train)))
            try:
                hessian_inv=np.linalg.inv(hessian)
            except :
                print('Non invertible')
                break
            theta=theta-np.dot(hessian_inv,dj)

    def predict(self,predict):
        x_pred=predict[:,:-1]
        y_pred=predict[:,-1]
        y_pred=np.reshape(y_pred,(y_pred.shape[0],1))
        predictions=[]
        for i in range(x_pred.shape[0]):
            predictions.append(np.dot(x_pred,self.theta))
        return predict
