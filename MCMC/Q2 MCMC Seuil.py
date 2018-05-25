import numpy as np
import numpy.random as npr
import  matplotlib.pyplot as plt

###############################################################################
## Q2. Splitting + MCMC : trouver les seuils à utiliser dans l'algorithme pour une valeur de P0, p, lambda, T
###############################################################################
plt.close()

P0 = 35
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

## Paramètre seuil
p = 0.7             #taux de conservation des sauts pour le processus de Markov
ratio = 0.1         #objectif pour les seuils : P(X in A_(k+1)|X in A_k) = ratio
n = 1000             #nombre de simulations
niveau = 0          # On cherche à calculer P(min P_t < niveau)

## Fonction auxilliaire 
# à partir de 2 array de temps de sauts t1 et t2 et 2 arrays de taille de sauts j1 et j2 -> retourne l'array des temps de sauts ordonnés et la l'array des valeurs des sauts associés.

def fusion(t1,j1,t2,j2):
    newT, newJ = np.zeros(t1.size+t2.size), np.zeros(t1.size+t2.size)
    i, j=0, 0
    s=0
    while(i<t1.size or j<t2.size):
        if (i == t1.size):
            newT[s:]=t2[j:]
            newJ[s:]=j2[j:]
            j=t2.size
        elif(j == t2.size):
            newT[s:]=t1[i:]
            newJ[s:]=j1[i:]
            i=t1.size
        else:
            if(t1[i] > t2[j]):
                newT[s]=t2[j]
                newJ[s]=j2[j]
                j+=1
            else:
                newT[s]=t1[i]
                newJ[s]=j1[i]
                i+=1
        s+=1
    return([newT,newJ])
    
# Génération d'un array de taille n dont les éléments alternent entre j0 et -j0
def oscillator(j0,n):
    return np.array([j0*(-1)**i for i in np.arange(n)])
    
##----------
# NIVEAU 1
##----------

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

# Premier seuil a1
a = [np.sort(minP)[int(ratio*n)]]

# Valeur initiale de la chaîne
argmin = np.argwhere(minP<=a[-1])                                     # indice des chaînes potentielles
index = npr.choice(argmin.reshape(argmin.size))               # choix au hasard d'un indice
TimeJump1 = TimeJump1[index]                                  # Temps de sauts (1)
TimeJump2 = TimeJump2[index]                                  # Temps de sauts (2)
JumpSize1 = JumpSize1[interval1[index]:interval1[index+1]]    # Taille de sauts
JumpSize2 = JumpSize2[index]                                  # Taille de sauts

PathPoissonInit=[TimeJump1,JumpSize1,TimeJump2,JumpSize2]             # Chaîne initiale


## BOUCLE pour les AUTRES NIVEAUX

while(a[-1] > niveau-1):
    PathPoisson = PathPoissonInit
    TimeJumps1 = []
    JumpSizes1 = []
    
    # Génération des chaînes (2) 
    NbrJump2 = npr.poisson(T*la2,n)
    TimeJump2 = T*np.random.uniform(0,1,size=np.sum(NbrJump2))
    JumpSize2 = npr.choice([-1,1],size=n)
    JumpSize2 = [oscillator(JumpSize2[i],NbrJump2[i]) for i in np.arange(n)]
    interval2 = np.cumsum(np.concatenate([np.arange(1), NbrJump2]))
    
    minP = np.zeros(n)
    for n_chain in np.arange(n):
        TimeJumps1.append(PathPoisson[0])
        JumpSizes1.append(PathPoisson[1])
        minP[n_chain] = min(P0+np.concatenate([np.arange(1),np.cumsum(fusion(TimeJumps1[-1], JumpSizes1[-1], np.sort(TimeJump2[interval2[n_chain]:interval2[n_chain+1]]), JumpSize2[n_chain])[1])]))
        
        # Quels sauts sont conserves
        Conserve=np.random.uniform(0,1,size=PathPoisson[0].size)<=p
        JumpTimeConserve = PathPoisson[0][Conserve]
        JumpSizeConserve=PathPoisson[1][Conserve]
        
        # Combien en ajoute-t-on
        NbrAjoute = np.random.poisson((1-p)*la1*T)
        
        # Instants de sauts ajoutés
        NewTimeJump = np.sort(T*np.random.uniform(0,1,NbrAjoute))
        NewJumpSize = npr.choice(val,size=NbrAjoute,p=prob)
        
        # Processus candidat à être la nouvelle valeur de la chaine
        NewPathPoisson=fusion(JumpTimeConserve,JumpSizeConserve,NewTimeJump,NewJumpSize)
                
        # Acceptation-rejet de ce candidat
        NewPrice = min(P0+np.concatenate([np.arange(1),np.cumsum(fusion(NewPathPoisson[0], NewPathPoisson[1], np.sort(TimeJump2[interval2[n_chain]:interval2[n_chain+1]]), JumpSize2[n_chain])[1])]))
        if NewPrice<=a[-1]:    # on accepte
            NewPathPoisson.extend([np.sort(TimeJump2[interval2[n_chain]:interval2[n_chain+1]]), JumpSize2[n_chain]])   
            PathPoisson = NewPathPoisson # mise à jour de la chaine
        
    a.append(np.sort(minP)[int(ratio*n)])

    # Valeur initiale de la chaîne
    argmin=np.argwhere(minP<=a[-1])                                         # indice des chaînes potentielles
    index=npr.choice(argmin.reshape(argmin.size))                           # choix au hasard d'un indice
    TimeJump1=TimeJumps1[index]                                             # Temps de sauts 
    TimeJump2 = np.sort(TimeJump2[interval2[index]:interval2[index+1]])     # Temps de sauts (2)
    JumpSize1=JumpSizes1[index]                                             # Taille de sauts
    JumpSize2 = JumpSize2[index]                                            # Taille de sauts

    PathPoissonInit=[TimeJump1, JumpSize1, TimeJump2, JumpSize2]            # Chaîne initiale

a[-1]=niveau-1

print("Valeur des seuils : {}".format(a))

