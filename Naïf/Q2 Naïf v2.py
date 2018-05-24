import numpy as np
import numpy.random as npr
import  matplotlib.pyplot as plt
import time
###############################################################################
## Q2. Splitting + MCMC pour un seul run de l'algo et une seule valeur de p
###############################################################################
plt.close()

P0 = 10
niveau = 0
T = 4*3600
la1 = 1/660
la2 = 1/110

N = [1,3]
P = [[1/2], [1/4,1/6,1/12]]

i=0

m = N[i]
p = P[i]

val = np.delete(np.arange(-m,m+1),m)
prob = np.concatenate([p[::-1],p])

n=int(1e4)

## Fonction auxilliaire 
# à partir de 2 array de temps de sauts t1 et t2 et 2 arrays de taille de sauts j1 et j2 -> retourne l'array des temps de sauts ordonnés et la l'array des valeurs des sauts associés.

def fusion(t1,j1,t2,j2):
    newT, newJ = np.zeros(t1.size+t2.size), np.zeros(t1.size+t2.size)
    i, j=0, 0
    s=0
    while(i<t1.size or j<t2.size):
        if (i == t1.size):
            newT[s:] = t2[j:]
            newJ[s:] = j2[j:]
            j=t2.size
        elif(j == t2.size):
            newT[s:] = t1[i:]
            newJ[s:] = j1[i:]
            i=t1.size
        else:
            if(t1[i] > t2[j]):
                newT[s] = t2[j]
                newJ[s] = j2[j]
                j+=1
            else:
                newT[s] = t1[i]
                newJ[s] = j1[i]
                i+=1
        s+=1
    return([newT,newJ])

# Génération d'un array de taille n dont les éléments alternent entre j0 et -j0
def oscillator(j0,n):
    return np.array([j0*(-1)**i for i in np.arange(n)])

##----------
# NIVEAU 1
##----------
print("MC estimation P(min(P_t) < {}) avec {} simulations".format(niveau, n))

TempsDepart = time.time()

# Nombre de sauts sur [0,T], par points de la chaîne
NbrJump1 = npr.poisson(T*la1,n)
NbrJump2 = npr.poisson(T*la2,n)

# Taille des sauts pour toutes les chaînes (1)
JumpSize1 = npr.choice(val,size=np.sum(NbrJump1),p=prob)

# Génération des chaînes (2)
JumpSize2 = npr.choice([-1,1],size=n)
JumpSize2 = [oscillator(JumpSize2[i],NbrJump2[i]) for i in np.arange(n)]

# Temps de sauts sur [0,T], pour toutes les chaînes
interval1 = np.cumsum(np.concatenate([np.arange(1), NbrJump1]))
interval2 = np.cumsum(np.concatenate([np.arange(1), NbrJump2]))

TimeJump1 = T*np.random.uniform(0,1,size=np.sum(NbrJump1))
TimeJump2 = T*np.random.uniform(0,1,size=np.sum(NbrJump2))

TimeJump1 = [np.sort(TimeJump1[interval1[i]:interval1[i+1]]) for i in np.arange(n)]
TimeJump2 = [np.sort(TimeJump2[interval2[i]:interval2[i+1]]) for i in np.arange(n)]

# Calcul des Prix min pour les n chaînes
minP=np.array([min(P0+np.concatenate([np.arange(1),np.cumsum(fusion(TimeJump1[i], JumpSize1[interval1[i]:interval1[i+1]], TimeJump2[i], JumpSize2[i])[1])])) for i in np.arange(n)])

#Affichage de l'estimateur de la proba pour ce niveau
ProbaEnd = np.sum(minP < niveau)/n

print("\n Durée d'exécution "+str(time.time()-TempsDepart))
print ("La proba estimée est " + str(ProbaEnd))