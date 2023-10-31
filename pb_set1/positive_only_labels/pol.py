import numpy as np
class PositiveOnlyLabels():
    def __init__(self):
        self.theta=0
    #Sigmoid function 
    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    def fit(self,x,lr=0.01,iterations=30):
        #Initializing variables
        x_train=x[:,:-1]
        intercept=np.ones((x_train.shape[0],1))
        x_train=np.append(intercept,x_train,axis=1)
        self.theta=np.array([[0 for _ in range(x_train.shape[0])]]).T
        y_train=x[:,-1:]
        prev_log=0
        for i in range(iterations):
            #Hypothesis
            z=np.dot(x_train,self.theta)
            hyp=self.sigmoid(-z)
            #gradient of cost function
            dj=np.dot((hyp-y_train).T,x_train).T
            #parameters train , 'theta'
            self.theta=self.theta-lr*dj
            #Loglikelihood check , to exit loop
            loglikelihood=np.dot(y_train.T,np.log(hyp))+np.dot((1-y_train).T,np.log(1-hyp))
            if loglikelihood<prev_log:
                prev_log=loglikelihood
            else :
                break;
    #predictions
    def predict(self,x):
        #Adding intercept array of ones!
        intercept=np.ones((x.shape[0],1))
        x_pred=np.append(intercept,x,axis=1)
        z=np.dot(x_pred,self.theta)
        #Returning values as and array of size mxn  ,{m number of training exp , n number of features}
        return self.sigmoid(-z)



