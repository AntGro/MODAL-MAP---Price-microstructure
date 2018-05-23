import numpy as np
import numpy.random as npr
import  matplotlib.pyplot as plt
import time

###############################################################################
## Splitting + MCMC pour un seul run de l'algo et une seule valeur de p
###############################################################################
plt.close()

P0 = 35
T=4*3600
la = 1/300

N=[1,3]
P=[[1/2],[1/4,1/6,1/12]]

i=0

m=N[i]
p=P[i]

val=np.delete(np.arange(-m,m+1),m)
prob=np.concatenate([p[::-1],p])

## Paramètre seuil
K=6
#p,M=0.1,int(8e4)
#p,M=0.5,int(1e4)
p,M=0.7,int(8e4)

# calcul des seuils successifs
BoundSplit = P0*(1-((np.arange(K,dtype=float)+1)**(1/2)/K**(1/2)))-1
BoundSplit=[24, 18.0, 14.0, 8.0, 6.0, 3.0, 1.0, -1]
K=len(BoundSplit)
print("Les seuils successifs envisagés pour le splitting sont \t")
print(BoundSplit); print("\n")


TempsDepart = time.time()

print("Lorsque p = "+ str(p) +" et lambda = " + str(la))   
plt.clf()
##•

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
NbrJump = npr.poisson(lam=la*T,size=(1,M))
Frequence = np.zeros(M+1,dtype=float)
#Boucle pour simuler la chaine de Markov
print("\t Chaine 1 sur "+ str(K))

for n_chain in np.arange(M):
    # nombre de sauts du processus de Poisson courant
    jj = NbrJump[0,n_chain]
    TimeJump = np.sort(T*np.random.uniform(0,1,size=jj))
    JumpSize=npr.choice(val,size=jj,p=prob)
    Price=P0+np.cumsum(np.concatenate([np.arange(1),JumpSize]))
    #Reserve = C+Lambda*np.concatenate([np.arange(1),TimeJump])-alpha*np.arange(jj+1)
    # on met a jour l'estimateur
    if min(Price)<=BoundSplit[0]:
        # on augmente le compteur
        Frequence[n_chain+1] = Frequence[n_chain]+1
        # on stocke ce point de la chaîne comme possible point de départ pour la chaine suivante
        PathPoissonInit = [TimeJump,JumpSize,Price]
    else: 
        # on ne change pas le compteur
        Frequence[n_chain+1] = Frequence[n_chain]
#Affichage de l'estimateur de la proba pour ce niveau
ProbaEnd = Frequence[-1]/M
print ("\t Pour le niveau 1, la proba estimée est " + str(ProbaEnd))
# Visualisation de la consistance de l'estimateur
plt.figure(1)
plt.plot(Frequence/(np.arange(M+1)+1), label="niveau 1") 
#plt.show()

## BOUCLE pour les AUTRES NIVEAUX
for n_level in (1+np.arange(K-1)):
    print("\t Chaine "+ str(n_level+1) + " sur " +str(K))
    Frequence = np.zeros(M+1,dtype=float)
    RateAccept = np.zeros(M+1,dtype=float)
    PathPoisson = PathPoissonInit
    MinPath = np.zeros(M+1,dtype=float)
    MinPath[0] = min(PathPoisson[2])
    for n_chain in np.arange(M):
        # Nombre de sauts dans le processus courant
        J = len(PathPoisson[0])
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
        NewPathPoisson=fusion(JumpTimeConserve,JumpSizeConserve,NewTimeJump,NewJumpSize)
        NewPathPoisson.append(P0+np.cumsum(np.concatenate([np.arange(1),NewPathPoisson[1]])))
        # Acceptation-rejet de ce candidat
        NewPrice = NewPathPoisson[-1]
        if min(NewPrice)<=BoundSplit[n_level-1]:    # on accepte
            PathPoisson = NewPathPoisson    # mise à jour de la chaine
            MinPath[n_chain+1] = min(NewPrice)    # stockage valeur minimale
            RateAccept[n_chain+1] = RateAccept[n_chain]+1   # update du taux d'acceptation-rejet
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
plt.title("Consistance des estimateurs a chaque niveau pour p = "+ str(p) +" et lambda = " + str(la))
plt.legend(loc="best")
plt.ylabel("Probabilite conditionnelle au niveau k")
plt.grid()


plt.figure(2)
plt.title("Evolution du taux d'acceptation pour p = "+ str(p) +" et lambda = " + str(la))
plt.ylabel("Taux d'acceptation")
plt.legend(loc="best")
plt.grid()

plt.show()
   
# Calcul de l'estimateur 
print("\n La proba de ruine estimée est " + str(ProbaEnd) +"\n")
print("(un IC sera donné dans le programme suivant)")

