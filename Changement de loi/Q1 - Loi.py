import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as sps
import math

##
n = 50000
alpha = 0.95
P0 = 35
T = 4*3600
lamb = 1/300  #temps moyen entre deux sauts : 300s

M = [1,3]
P = [[1/2],[1/4,1/6,1/12]]

i = 0

m = M[i]
p = P[i]

val = np.delete(np.arange(-m,m+1),m)
prob = np.concatenate([p[::-1],p])

def f(x, theta):
    return theta*x

f= np.vectorize(f)


## Proba inf(P)<0

def process(k,theta,proba,pI=P0):
    J = npr.choice(val,size=k,p=proba)
    s = np.cumsum(np.concatenate([np.arange(1),J]))
    normalisation = np.exp(-np.sum(f(J,theta))+lamb*T*(np.dot(np.exp(f(val,theta)),prob)-1))
    return (min(s) < -pI)*normalisation

def estimation_proba(theta):
    lambNew = lamb*np.dot(np.exp(f(val,theta)),prob)
    probNew = np.multiply(np.exp(f(val,theta)),prob)
    probNew = probNew/(np.sum(probNew))
    X = npr.poisson(T*lambNew,n)
    return np.mean(np.array([process(k,theta,proba=probNew) for k in X]))

estimation_proba=np.vectorize(estimation_proba)

def plot():
    theta = np.linspace(-1.2,0,50)
    estimation = estimation_proba(theta)
    plt.plot(theta,estimation)
    plt.show()
    
def bestTheta(inf,sup,nbrPoints, M):
    theta=np.linspace(inf,sup,nbrPoints)
    estimations=[]
    bestStd=+math.inf
    bestTh=inf
    for th in theta:
        for k in range(M):
            estimations.append(estimation_proba(th))
        ecartType=np.std(np.array(estimations))
        if (ecartType<bestStd):
            bestStd=ecartTRype
            bestTh=th
    return bestTh

pEst = estimation_proba(-0.8)
sEst = pEst*(1-pEst)

qInf = sps.norm.ppf((1-alpha)/2)
qSup = sps.norm.ppf((1+alpha)/2)

bInf = pEst-qSup*sEst/np.sqrt(n)
bSup = pEst-qInf*sEst/np.sqrt(n)

## Quantile 

# ATTENTION; n simulations à chaque appel de pEst

def process2(k,theta,proba,c,pI=P0):
    J = npr.choice(val,size=k,p=proba)
    s = pI+np.sum(J)
    normalisation = np.exp(-np.sum(f(J,theta))+lamb*T*(np.dot(np.exp(f(val,theta)),prob)-1))
    return (s < c)*normalisation

def estimation_proba2(theta,c):
    lambNew = lamb*np.dot(np.exp(f(val,theta)),prob)
    probNew = np.multiply(np.exp(f(val,theta)),prob)
    probNew = probNew/(np.sum(probNew))
    X = npr.poisson(T*lambNew,n)
    return np.mean(np.array([process2(k,theta,probNew,c) for k in X]))

def dichotomie(pEst,seuil, sup, theta):
    pas=1
    inf=sup-1
    
    while (pEst(theta,inf)>seuil):
        pas=2*pas
        inf=inf-pas
    
    while(inf<sup):
        c = (sup+inf)//2
        if (pEst(theta,c)<seuil):
            inf = c
        else:
            sup = c
    return inf

#theta = np.linspace(-1.2,0,50)
#dich=np.array([dichotomie(estimation_proba2,0.01,5,th) for th in theta])
#plt.plot(theta,dich)
#plt.show()

## Quantile 2

def process3(seuil,theta):
    res = []
    sIS = []
    
    lambNew = lamb*np.dot(np.exp(f(val,theta)),prob)
    probNew = np.multiply(np.exp(f(val,theta)),prob)
    probNew = probNew/(np.sum(probNew))
    X = npr.poisson(T*lambNew,n)
    
    for i in range(n):
        J = npr.choice(val,size=X[i],p=probNew)
        s = P0+np.sum(J)
        normalisation = np.exp(-np.sum(f(J,theta))-lamb*T*(np.dot(np.exp(f(val,theta)),prob)-1))
        res.append(s)
        sIS.append(normalisation)
    res=np.array(res)
    sIS=np.array(sIS)
    indexSort=np.argsort(res)
    sIsSort=sIS[indexSort]
    #print(sIsSort)
    Y=np.cumsum(sIsSort)
    index=np.argwhere(Y>=n*seuil)[0][0]
    print(index)

    return res[indexSort[index]]

theta=np.linspace(-0.25,-0.05,15)
y=[process3(0.0000001,th) for th in theta]
plt.plot(theta,y)
plt.show()

