import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as sps

##
n = 10000
alpha = 0.95
P0 = 20
T = 4*3600
lamb = 1/300  #temps moyen entre deux sauts : 300s


M = [1,3]
P = [[1/2],[1/4,1/6,1/12]]

i = 0

m = M[i]
p = P[i]

val = np.delete(np.arange(-m,m+1),m)
prob = np.concatenate([p[::-1],p])

def f(x, theta = 0.1):
    return theta*x

f=np.vectorize(f)

lambNew = lamb*np.dot(np.exp(f(val)),prob)
probNew = np.multiply(np.exp(f(val)),prob)
probNew = probNew/(np.sum(probNew))

## Proba inf(P)<0

X = npr.poisson(T*lambNew,n)

def process(k,pI=P0,distr=[val,probNew]):
    J = npr.choice(distr[0],size=k,p=distr[1])
    s = np.cumsum(J)
    normalisation = np.exp(-np.sum(f(J))+lambNew*T*(np.dot(np.exp(f(val)),prob)-1))
    return (min(s) < -pI)*normalisation

process = np.vectorize(process)

pEst = np.mean(process(X))


