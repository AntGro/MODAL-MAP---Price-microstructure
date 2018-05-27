import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as sps
import time

##

n = int(1e4)
P0 = 10
niveau=2
T = 4*3600
lamb = 300  #temps moyen entre deux sauts : 300s
alpha = -0.875
m = 3
proba = [1/2,3/8,1/8]
confiance=0.95

val = np.delete(np.arange(-m,m+1),m)


#Calcul de sign(J)
def signJ(k):
    res = [npr.choice([1,-1],p=[(1+alpha)/2,(1-alpha)/2])] #initialisation non précisée
    for i in range(1,k):
        res.append(res[-1]*npr.choice([1,-1],p=[(1+alpha)/2,(1-alpha)/2]))
    return res
    
##Vérification  que P(sgn(Jn)*sgn(Jn+1))=(1+alpha)/2

taille = 10000
echantillon = []
for i in range(taille):
    L=signJ(3)
    echantillon.append(L[1]*L[2])

probabilite=np.sum(np.array(echantillon)==1)/taille

##Probability inf(P)<0
def process():
    P=P0+np.cumsum(signJ(npr.poisson(T/lamb)))
    return min(P)<niveau

TempsDepart = time.time()

pEst=np.mean(np.array([process() for k in range(n)]))

print("\nDurée d'exécution "+str(time.time()-TempsDepart))
print ("La proba estimée est " + str(pEst))


sEst=pEst*(1-pEst)

qInf=sps.norm.ppf((1-confiance)/2)
qSup=sps.norm.ppf((1+confiance)/2)

bInf=pEst-qSup*sEst/np.sqrt(n)
bSup=pEst-qInf*sEst/np.sqrt(n)


##Quantiles à q1

q1 = 1e-4
q2 = 1-q1

X = npr.poisson(T/lamb,n)
Z = P0+np.array([np.sum(np.multiply(signJ(k),npr.choice(1+np.arange(m),p=proba))) for k in X])
Z = np.sort(Z)

print(Z[int(q1*n)],Z[int(q2*n)])

plt.hist(Z,np.arange(min(Z),max(Z)+1)-0.5,normed = True)
plt.plot(np.linspace(min(Z),max(Z)),sps.norm(P0,np.std(Z)).pdf(np.linspace(min(Z),max(Z))))
plt.show()