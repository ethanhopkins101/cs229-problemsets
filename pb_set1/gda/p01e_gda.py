import numpy as np
from math import pi
from math import sqrt
class Gda():
    def __init__(self):
        self.feta=self.mu0=self.mu1=self.sigma=0


    def fit(self, x, y):
        x0=x[y[:,0]==0]
        x1=x[y[:,0]==1]
        m=x.shape[0]
        self.feta=np.mean(y)
        self.mu0=np.mean(x0,axis=0)
        self.mu1=np.mean(x1,axis=0)
        ## not sure of the next syntax
        mu_vec=np.where(y==1,self.mu1,self.mu0)
        self.sigma=(1/m)*np.dot((x-mu_vec).T,x-mu_vec)

    def predict(self, x):
         #           p(x/y=1)p(y=1)                      p(x/y=0)p(y)
    # p(y=1|x)=       ____________            p(y=0|x) =  __________
    #                     p(x)                                p(x)
        px_y0=1/(((2*pi)**(x.shape[1]/2))*sqrt(np.linalg.det(self.sigma)))*np.exp(-(np.dot(np.dot(x-self.mu0,np.linalg.inv(self.sigma)),(x-self.mu0).T)/2))
        px_y1=1/(((2*pi)**(x.shape[1]/2))*sqrt(np.linalg.det(self.sigma)))*np.exp(-(np.dot(np.dot(x-self.mu1,np.linalg.inv(self.sigma)),(x-self.mu1).T)/2))
        py1=self.feta
        py0=1-self.feta
        px=px_y0*py0+px_y1*py1
        py0_x=px_y0*py0/px
        py1_x=px_y1*py1/px
        return max(py1_x,py0_x)
