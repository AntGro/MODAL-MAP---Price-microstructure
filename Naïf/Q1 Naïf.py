import numpy as np
import numpy.random as npr
import scipy.stats as sps

##
n=10000
alpha=0.95
P0=10
T=4*3600
lamb=660  #temps moyen entre deux sauts : 300s

M=[1,3]
P=[[1/2],[1/4,1/6,1/12]]

i=0

m=M[i]
p=P[i]

val=np.delete(np.arange(-m,m+1),m)
prob=np.concatenate([p[::-1],p])

X=npr.poisson(T/lamb,n)

def process(k,pI=P0,distr=[val,prob]):
    s=np.cumsum(npr.choice(distr[0],size=k,p=distr[1]))
    return(min(s)<-pI)
   
process=np.vectorize(process)

pEst=np.mean(process(X))
sEst=pEst*(1-pEst)

qInf=sps.norm.ppf((1-alpha)/2)
qSup=sps.norm.ppf((1+alpha)/2)

bInf=pEst-qSup*sEst/np.sqrt(n)
bSup=pEst-qInf*sEst/np.sqrt(n)

print("La probabilité associée à la distribution {0} avec P0={1} est estimée à : \n {2} comprise entre {3:6f} et {4:6f} au niveau de confiance {5}".format([list(val),list(prob)],P0,pEst,bInf,bSup,alpha))

##
n=int(1e5)
P0=10
T=4*3600
lamb=660  #temps moyen entre deux sauts : 300s

q1=1e-4
q2=1-q1

M=[1,3]
P=[[1/2],[1/4,1/6,1/12]]

i=0

m=M[i]
p=P[i]

val=np.delete(np.arange(-m,m+1),m)
prob=np.concatenate([p[::-1],p])

X=npr.poisson(T/lamb,n)

Z=P0+np.array([np.sum(npr.choice(val,size=k,p=prob)) for k in X])
Z=np.sort(Z)

print(Z[int(q1*n)],Z[int(q2*n)])

