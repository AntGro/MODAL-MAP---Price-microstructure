import numpy as np
import numpy.random as npr
import scipy.stats as sps
import time
###############################################################################
## Q1. Naïf
###############################################################################
P0 = 35
niveau = 0

T = 4*3600
la = 1/300

N = [1,3]
P = [[1/2], [1/4,1/6,1/12]]

i=1

m = N[i]
p = P[i]

val = np.delete(np.arange(-m,m+1),m)
prob = np.concatenate([p[::-1],p])

n=int(1e5)
alpha=0.95


##----------
# Estimation de la probabilité d'avoir un prix négatif
##----------
print("MC estimation P(min(P_t) < {}) avec {} simulations".format(niveau, n))

TempsDepart = time.time()

# Nombre de sauts sur [0,T], par points de la chaîne
NbrJump = npr.poisson(T*la,n)

# Taille des sauts pour toutes les chaînes 
JumpSize = npr.choice(val,size=np.sum(NbrJump),p=prob)

# Découpage des chaînes
interval = np.cumsum(np.concatenate([np.arange(1), NbrJump]))
   
# Calcul des Prix min pour les n chaînes
minP=np.array([min(P0+np.concatenate([np.arange(1),np.cumsum(JumpSize[interval[i]:interval[i+1]])])) for i in np.arange(n)])

#Affichage de l'estimateur de la proba pour ce niveau
pEst = np.sum(minP < niveau)/n

print("\nDurée d'exécution "+str(time.time()-TempsDepart))
print ("La proba estimée est " + str(pEst))

sEst=pEst*(1-pEst)

qInf=sps.norm.ppf((1-alpha)/2)
qSup=sps.norm.ppf((1+alpha)/2)

bInf=pEst-qSup*sEst/np.sqrt(n)
bSup=pEst-qInf*sEst/np.sqrt(n)

print("La probabilité associée à la distribution {0} avec P0={1} est estimée à : \n {2} comprise entre {3:6f} et {4:6f} au niveau de confiance {5}".format([list(val),list(prob)],P0,pEst,bInf,bSup,alpha))

##----------
# Estimation des quartiles
##----------
# q1=1e-7
# q2=1-q1
# 
# print("MC estimation quartile au niveau {} avec {} simulations".format(q1, n))
# 
# TempsDepart = time.time()
# 
# # Nombre de sauts sur [0,T], par points de la chaîne
# NbrJump = npr.poisson(T*la,n)
# 
# Pt=P0+np.array([np.sum(npr.choice(val,size=NbrJump[i],p=prob)) for i in np.arange(n)])
# distrib=np.sort(Pt)
# 
# print("\n Durée d'exécution "+str(time.time()-TempsDepart))
# print(distrib[int(q1*n)],distrib[int(q2*n)])
