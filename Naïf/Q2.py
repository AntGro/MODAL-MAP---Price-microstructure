import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as sps
import time
##

n=int(1000)
P0=10
T=4*3600
lamb1=660  #temps moyen entre deux sauts du processus 1 : 660s
lamb2=110 #temps moyen entre deux sauts du processus 2 : 110s

confiance=0.95


q1=1e-4
q2=1-q1

M=[1,3]
P=[[1/2],[1/4,1/6,1/12]]

i=0

m=M[i]
p=P[i]

val=np.delete(np.arange(-m,m+1),m)
prob=np.concatenate([p[::-1],p])


##Probability inf(P)<0
def process():
    res=P0
    NT1=npr.poisson(T/lamb1)
    instantSaut1=np.sort(npr.uniform(0,T,NT1))
    NT2=npr.poisson(T/lamb2)
    instantSaut2=np.sort(npr.uniform(0,T,NT2))
    index=npr.choice([-1,1])
    i1=0
    i2=0
    while i1<len(instantSaut1) and i2<len(instantSaut2):
        a,b=instantSaut1[i1],instantSaut2[i2]
        if a<b:
            res=res+npr.choice(val,p=prob)
            if res<0:
                return 1
            i1+=1
        else:
            res=res+index
            if res<0:
                return 1
            index=-index
            i2+=1
    if i1==len(instantSaut1):
        while i2<len(instantSaut2):
            res=res+index
            if res<0:
                return 1
            index=-index
            i2+=1
    if i2==len(instantSaut2):
        while i1<len(instantSaut1):
            res=res+npr.choice(val,p=prob)
            if res<0:
                return 1
            i1+=1
    return 0

TempsDepart = time.time()


pEst=np.mean(np.array([process() for k in range(n)]))

print(time.time()-TempsDepart)
print(pEst)


sEst=pEst*(1-pEst)

qInf=sps.norm.ppf((1-confiance)/2)
qSup=sps.norm.ppf((1+confiance)/2)

bInf=pEst-qSup*sEst/np.sqrt(n)
bSup=pEst-qInf*sEst/np.sqrt(n)


##Quantile à q1

q1=1e-4
q2=1-q1

X1=npr.poisson(T/lamb1,n)
Z1=(P0/2)+np.array([np.sum(npr.choice(val,size=k,p=prob)) for k in X1])

X2=npr.poisson(T/lamb2,n)
Z2=(P0/2)+np.array([np.sum([(-1)**(i+npr.randint(2)) for i in range(k)]) for k in X2])

Z=Z1+Z2
Z=np.sort(Z)

print(Z[int(q1*n)],Z[int(q2*n)])

plt.hist(Z,np.arange(min(Z),max(Z)+1)-0.5,normed=True)
plt.plot(np.linspace(min(Z),max(Z)),sps.norm(P0,np.std(Z)).pdf(np.linspace(min(Z),max(Z))))
plt.show()






    

