import numpy as np
import numpy.random as npr
import  matplotlib.pyplot as plt
import time
###############################################################################
## Q2. Splitting + MCMC pour un seul run de l'algo et une seule valeur de p
###############################################################################
plt.close()

P0 = 10
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
#p,M=0.1,int(8e4)
#p,M=0.5,int(1e4)
p,M=0.7,int(8e4)

# Seuils successif (calculé avec le programme précédent)
BoundSplit = [2.0, -3.0, -7]
K = len(BoundSplit)

print("Les seuils successifs envisagés pour le splitting sont \t")
print(BoundSplit); print("\n")

TempsDepart = time.time()

print("Lorsque p = "+ str(p) +" et lambda1 = " + str(la1) + ", lambda2 = "+str(la2))   
plt.clf()

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
print("\t Chaine 1 sur "+ str(K))

# Nombre de sauts sur [0,T], par points de la chaîne
NbrJump1 = npr.poisson(T*la1,M)
NbrJump2 = npr.poisson(T*la2,M)

# Taille des sauts pour toutes les chaînes (1)
JumpSize1 = npr.choice(val,size=np.sum(NbrJump1),p=prob)

# Génération des chaînes (2)
JumpSize2 = npr.choice([-1,1],size=M)
JumpSize2 = [oscillator(JumpSize2[i],NbrJump2[i]) for i in np.arange(M)]

# Temps de sauts sur [0,T], pour toutes les chaînes
interval1 = np.cumsum(np.concatenate([np.arange(1), NbrJump1]))
interval2 = np.cumsum(np.concatenate([np.arange(1), NbrJump2]))

TimeJump1 = T*np.random.uniform(0,1,size=np.sum(NbrJump1))
TimeJump2 = T*np.random.uniform(0,1,size=np.sum(NbrJump2))

TimeJump1 = [np.sort(TimeJump1[interval1[i]:interval1[i+1]]) for i in np.arange(M)]
TimeJump2 = [np.sort(TimeJump2[interval2[i]:interval2[i+1]]) for i in np.arange(M)]

# Calcul des Prix min pour les M chaînes
minP=np.array([min(P0+np.concatenate([np.arange(1),np.cumsum(fusion(TimeJump1[i], JumpSize1[interval1[i]:interval1[i+1]], TimeJump2[i], JumpSize2[i])[1])])) for i in np.arange(M)])

# Valeur initiale de la chaîne
argmin = np.argwhere(minP<=BoundSplit[0])                     # indice des chaînes potentielles
index = npr.choice(argmin.reshape(argmin.size))               # choix au hasard d'un indice
TimeJump1 = TimeJump1[index]                                  # Temps de sauts (1)
TimeJump2 = TimeJump2[index]                                  # Temps de sauts (2)
JumpSize1 = JumpSize1[interval1[index]:interval1[index+1]]    # Taille de sauts
JumpSize2 = JumpSize2[index]                                  # Taille de sauts

PathPoissonInit = [TimeJump1,JumpSize1,TimeJump2,JumpSize2]     # Chaîne initiale

#Affichage de l'estimateur de la proba pour ce niveau
Frequence = np.concatenate([np.arange(1),np.cumsum(minP<=BoundSplit[0])/np.arange(1,M+1)])
ProbaEnd = Frequence[-1]
print ("\t Pour le niveau 1, la proba estimée est " + str(ProbaEnd))

# Visualisation de la consistance de l'estimateur
plt.figure(1)
plt.plot(Frequence, label="niveau 1") 

## BOUCLE pour les AUTRES NIVEAUX
for n_level in (1+np.arange(K-1)):
    print("\t Chaine "+ str(n_level+1) + " sur " +str(K))
    PathPoisson = PathPoissonInit
        
    # Génération des chaînes (2) 
    NbrJump2 = npr.poisson(T*la2,M)
    JumpSize2 = npr.choice([-1,1],size=M)
    JumpSize2 = [oscillator(JumpSize2[i],NbrJump2[i]) for i in np.arange(M)]
    interval2 = np.cumsum(np.concatenate([np.arange(1), NbrJump2]))
    TimeJump2 = T*np.random.uniform(0,1,size=np.sum(NbrJump2))
    TimeJump2 = [np.sort(TimeJump2[interval2[i]:interval2[i+1]]) for i in np.arange(M)]

    Frequence = np.zeros(M+1,dtype=float)
    RateAccept = np.zeros(M+1,dtype=float)
    MinPath = np.zeros(M+1,dtype=float)
    MinPath[0] = minP[index]
    
    for n_chain in np.arange(M):        
        # Quels sauts sont conserves dans la première chaîne
        Conserve=np.random.uniform(0,1,size=PathPoisson[0].size)<=p
        JumpTimeConserve = PathPoisson[0][Conserve]
        JumpSizeConserve=PathPoisson[1][Conserve]
        
        # Combien en ajoute-t-on dans la première chaîne
        NbrAjoute = np.random.poisson((1-p)*la1*T)
        
        # Instants et tailles de sauts ajoutés dans la première chaîne
        NewTimeJump = np.sort(T*np.random.uniform(0,1,NbrAjoute))
        NewJumpSize = npr.choice(val,size=NbrAjoute,p=prob)
        
        # Processus candidat à être la nouvelle valeur de la chaine
        NewPathPoisson=fusion(JumpTimeConserve,JumpSizeConserve,NewTimeJump,NewJumpSize)
                
        # Acceptation-rejet de ce candidat
        NewPrice = min(P0+np.concatenate([np.arange(1),np.cumsum(fusion(NewPathPoisson[0], NewPathPoisson[1], TimeJump2[n_chain], JumpSize2[n_chain])[1])]))
        
        if NewPrice <= BoundSplit[n_level-1]:    # on accepte
            RateAccept[n_chain+1] = RateAccept[n_chain]+1   # update du taux d'acceptation-rejet
            MinPath[n_chain+1]=NewPrice
            NewPathPoisson.extend([TimeJump2[n_chain], JumpSize2[n_chain]])   
            PathPoisson = NewPathPoisson # mise à jour de la chaine
        else:       # on refuse
            MinPath[n_chain+1] = MinPath[n_chain]   #   stockage valeur minimale
            RateAccept[n_chain+1] = RateAccept[n_chain] # update du taux d'acceptation-rejet
        
        # Calcul de l'estimateur de la probabilité
        if MinPath[n_chain+1]<=BoundSplit[n_level]:
            Frequence[n_chain+1] = Frequence[n_chain]+1
            PathPoissonInit = PathPoisson
        else:
            Frequence[n_chain+1] = Frequence[n_chain]
    #
    # Affichage
    #
    NewProbaEnd = Frequence[-1]/M
    ProbaEnd = ProbaEnd*NewProbaEnd
    print ("\t Pour le niveau " + str(n_level+1) +", la proba estimée est " + str(NewProbaEnd))    
    # Visualisation de la consistance de l'estimateur
    plt.figure(1)
    plt.plot(Frequence/(np.arange(M+1,dtype=float)+1), label="niveau %1.0f" %(n_level+1))  
           
    # Visualisation de l'évolution du taux d'acceptation-rejet
    plt.figure(2)
    plt.plot(RateAccept/(np.arange(M+1,dtype=float)+1), label="niveau %1.0f" %(n_level+1)) 

TempsFin = time.time()
print("\n Durée d'exécution "+str(TempsFin-TempsDepart))

plt.figure(1)
plt.title("Consistance des estimateurs a chaque niveau pour p = "+ str(p) +" et lambda = " + str(la1))
plt.legend(loc="best")
plt.ylabel("Probabilite conditionnelle au niveau k")
plt.grid()


plt.figure(2)
plt.title("Evolution du taux d'acceptation pour p = "+ str(p) +" et lambda = " + str(la1))
plt.ylabel("Taux d'acceptation")
plt.legend(loc="best")
plt.grid()

plt.show()
   
# Calcul de l'estimateur 
print("\n La proba de ruine estimée est " + str(ProbaEnd) +"\n")
print("(un IC sera donné dans le programme suivant)")
