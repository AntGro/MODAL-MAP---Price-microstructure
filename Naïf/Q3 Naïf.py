import numpy as np
import numpy.random as npr
import scipy.stats as sps
import time

###############################################################################
## Q1. Naïf
###############################################################################
P0 = 10
niveau = 10

T = 4*3600
la = 1/300

N = [1,3]
P = [[1], [1/2,3/8,1/8]]

i=0

m = N[i]
p = P[i]

val = np.arange(1,m+1)
prob = p

n=int(1e4)
alpha = -0.875

confiance = 0.95    

##----------
# Estimation de la probabilité d'avoir un prix négatif
##----------

def generateSign(rand,a=alpha):
    s=[2*(rand[0]<0.5)-1]
    for i in np.arange(1,rand.size):
        s.append((2*(rand[i]<((1+s[-1]*a)/2))-1))
    return np.array(s)


TempsDepart = time.time()

# Nombre de sauts sur [0,T], par points de la chaîne
NbrJump = npr.poisson(T*la,n)

# Découpage des chaînes
interval = np.cumsum(np.concatenate([np.arange(1), NbrJump]))

# Taille des sauts pour toutes les chaînes 
JumpSizeAbs = npr.choice(val,size=np.sum(NbrJump),p=prob)

# Signe des sauts
randTransition=npr.rand(np.sum(NbrJump))
JumpSign = [generateSign(randTransition[interval[i]:interval[i+1]],alpha) for i in np.arange(n)]
   
# Calcul des Prix min pour les n chaînes
minP=np.array([min(P0+np.concatenate([np.arange(1),np.cumsum(JumpSizeAbs[interval[i]:interval[i+1]]*JumpSign[i])])) for i in np.arange(n)])

#Affichage de l'estimateur de la proba pour ce niveau
pEst = np.mean(minP < niveau)

print("\nDurée d'exécution "+str(time.time()-TempsDepart))
print ("La proba estimée est " + str(pEst))


#plt.hist(minP,np.arange(min(minP),P0+1),normed=True,cumulative=True)
#plt.show()

#print(np.sum(np.array([np.mean(JumpSign[i][:-1]*JumpSign[i][1:]==1)*NbrJump[i] for i in range(n)]))/np.sum(NbrJump))

sEst=pEst*(1-pEst)

qInf=sps.norm.ppf((1-alpha)/2)
qSup=sps.norm.ppf((1+alpha)/2)

bInf=pEst-qSup*sEst/np.sqrt(n)
bSup=pEst-qInf*sEst/np.sqrt(n)

##
q1=1e-4
q2=1-q1

print("MC estimation quartile au niveau {} avec {} simulations".format(q1, n))

TempsDepart = time.time()

# Nombre de sauts sur [0,T], par points de la chaîne
NbrJump = npr.poisson(T*la,n)

# Découpage des chaînes
interval = np.cumsum(np.concatenate([np.arange(1), NbrJump]))

# Signe des sauts
randTransition=npr.rand(np.sum(NbrJump))
JumpSign = [generateSign(randTransition[interval[i]:interval[i+1]],alpha) for i in np.arange(n)]

Pt=P0+np.array([np.sum(npr.choice(val,size=NbrJump[i],p=prob)*JumpSign[i]) for i in np.arange(n)])
distrib=np.sort(Pt)

print("\nDurée d'exécution "+str(time.time()-TempsDepart))
print(distrib[int(q1*n)],distrib[int(q2*n)])