import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

###############################################################################
## Splitting + MCMC : trouver les seuils à utiliser dans l'algorithme pour une valeur de P0, p, lambda, T
###############################################################################
plt.close()

P0 = 35
T = 4*3600
la = 1/300

N = [1,3]
P = [[1/2],[1/4,1/6,1/12]]

i = 0

m = N[i]
p = P[i]

val = np.delete(np.arange(-m,m+1),m)
prob = np.concatenate([p[::-1],p])

## Paramètre seuil
p = 0.7             #taux de conservation des sauts pour le processus de Markov
ratio = 0.1         #objectif pour les seuils : P(X in A_(k+1)|X in A_k) = ratio
n = int(1e4)        #nombre de simulations
niveau = 0          # On cherche à calculer P(min P_t < niveau)

## Fonction auxilliaire 
def fusion(t1,j1,t2,j2):
    newT,newJ=np.zeros(t1.size+t2.size),np.zeros(t1.size+t2.size)
    i,j=0,0
    s=0
    while(i<t1.size or j<t2.size):
        if (i==t1.size):
            newT[s:]=t2[j:]
            newJ[s:]=j2[j:]
            j=t2.size
        elif(j==t2.size):
            newT[s:]=t1[i:]
            newJ[s:]=j1[i:]
            i=t1.size
        else:
            if(t1[i]>t2[j]):
                newT[s]=t2[j]
                newJ[s]=j2[j]
                j+=1
            else:
                newT[s]=t1[i]
                newJ[s]=j1[i]
                i+=1
        s+=1
    return([newT,newJ])
    

##----------
# NIVEAU 1
##----------

# Nombre de sauts sur [0,T], par points de la chaîne
NbrJump=npr.poisson(T*la,n)

# Temps de sauts sur [0,T], pour toutes les chaînes
TimeJump = T*np.random.uniform(0,1,size=np.sum(NbrJump))

# Taille des sauts pour toutes les chaînes
JumpSize = npr.choice(val,size=np.sum(NbrJump),p=prob)

# Calcul des prix min pour chaque chaîne
interval = np.cumsum(np.concatenate([np.arange(1),NbrJump]))
minP = np.array([min(P0+np.concatenate([np.arange(1),np.cumsum(JumpSize[interval[i]:interval[i+1]])])) for i in np.arange(n)])

# Premier seuil a1
a = [np.sort(minP)[int(ratio*n)]]

# Valeur initiale de la chaîne
argmin = np.argwhere(minP<=a[-1])                               # indice des chaînes potentielles
index = npr.choice(argmin.reshape(argmin.size))                 # choix au hasard d'un indice
TimeJump = np.sort(TimeJump[interval[index]:interval[index+1]]) # Temps de sauts 
JumpSize = JumpSize[interval[index]:interval[index+1]]          # Taille de sauts

PathPoissonInit = [TimeJump,JumpSize]                           # Chaîne initiale


## BOUCLE pour les AUTRES NIVEAUX

while(a[-1] > niveau-1):
    PathPoisson = PathPoissonInit
    TimeJumps = []
    JumpSizes = []
    minP = np.zeros(n)
    for n_chain in np.arange(n):
        
        TimeJumps.append(PathPoisson[0])
        JumpSizes.append(PathPoisson[1])
        minP[n_chain] = min(P0+np.cumsum(np.concatenate([np.arange(0),PathPoisson[1]])))
        
        # Quels sauts sont conserves
        Conserve=np.random.uniform(0,1,size=PathPoisson[0].size)<=p
        JumpTimeConserve = PathPoisson[0][Conserve]
        JumpSizeConserve=PathPoisson[1][Conserve]
        
        # Combien en ajoute-t-on
        NbrAjoute = np.random.poisson((1-p)*la*T)
        
        # Instants de sauts ajoutés
        NewTimeJump = np.sort(T*np.random.uniform(0,1,NbrAjoute))
        NewJumpSize = npr.choice(val,size=NbrAjoute,p=prob)
        
        # Processus candidat à être la nouvelle valeur de la chaine
        NewPathPoisson = fusion(JumpTimeConserve,JumpSizeConserve,NewTimeJump,NewJumpSize)
        NewPrice = P0+np.cumsum(np.concatenate([np.arange(1),NewPathPoisson[1]]))
        
        # Acceptation-rejet de ce candidat
        if min(NewPrice)<=a[-1]:    # on accepte
            PathPoisson = NewPathPoisson    # mise à jour de la chaine
    
    # Valeur initiale de la chaîne  
    a.append(np.sort(minP)[int(ratio*n)])

    # Valeur initiale de la chaîne
    argmin=np.argwhere(minP<=a[-1]) # indice des chaînes potentielles
    index=npr.choice(argmin.reshape(argmin.size)) # choix au hasard d'un indice
    TimeJump=TimeJumps[index] # Temps de sauts 
    JumpSize=JumpSizes[index] # Taille de sauts

    PathPoissonInit=[TimeJump,JumpSize] # Chaîne initiale

a[-1]=niveau-1

print("Valeur des seuils : {}".format(a))

