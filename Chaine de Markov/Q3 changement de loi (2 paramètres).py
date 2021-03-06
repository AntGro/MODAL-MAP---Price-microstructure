import numpy as np
import numpy.random as npr
import scipy.stats as sps
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools


###############################################################################
## Q3. Changement de loi
###############################################################################
plt.close()

P0 = 10
niveau = 0

T = 4*3600
la = 1/300

N = [1,3]
P = [[1], [1/2,1/3,1/6]]

i=0

m = N[i]
p = P[i]

val = np.arange(1,m+1)
prob = p

n=int(1e4)
N = 100 # Nombre de sauts par points de la chaîne

# paramètre de la matrice de transition
alpha = -0.875

#Nouveaux paramètres de transition
thetaP = -0.875
thetaM = -0.875

confiance = 0.95    

##----------
# Estimation de la probabilité d'avoir un prix négatif
##----------

 #étant donné un array de nombres aléatoires 'rand' renvoie génère la chaîne de Markov des signes suivant la matrice de transition de paramètre alpha
def generateSign(rand,tP=thetaP, tM=thetaM):
    s=[2*(rand[0]<0.5)-1]
    for i in np.arange(1,rand.size):
        if(s[-1]==1):
            s.append((2*(rand[i]<((1+s[-1]*tP)/2))-1))
        else:
            s.append((2*(rand[i]<((1+s[-1]*tM)/2))-1))
        
    return np.array(s)

def Ln(x, y, a=alpha, tP=thetaP, tM=thetaM): # retourne le rapport de vraissemblance de P_a par rapport à P_t
    if (x == y):
        if(x==1):
            return(1+a)/(1+tP)
        return (1+a)/(1+tM)
    if(x==1):
        return (1-a)/(1-tP)
    return (1-a)/(1-tM)


## Simu script
TempsDepart = time.time()

#nombre total de sauts à simuler
NbrJump = n * N 

# Taille des sauts pour toutes les chaînes 
JumpSizeAbs = npr.choice(val,size=N*n,p=prob)

# Signe des sauts --- /!\ Sous theta /!\
randTransition=npr.rand(n,N)
JumpSign = np.apply_along_axis(generateSign, axis=1, arr=randTransition, tP=thetaP, tM=thetaM)

# Calcul des Prix min multipliés par Ln pour chacune des n chaînes 
minP=np.array([(min(P0+np.concatenate([np.arange(1),np.cumsum(JumpSizeAbs[i*N:i*N+N]*JumpSign[i])]))< niveau) * np.product(np.array([Ln(JumpSign[i,j],JumpSign[i,j+1]) for j in np.arange(N-1)])) for i in np.arange(0,n)])

#Affichage de l'estimateur de la proba pour ce niveau
pEst=np.mean(minP)


print("\nDurée d'exécution "+str(time.time()-TempsDepart))
print ("La proba estimée est " + str(pEst))

for i in range(5):
    plt.step(np.arange(N+1),P0+np.concatenate([np.arange(1),np.cumsum(JumpSizeAbs[i*N:i*N+N]*JumpSign[i])]))
    plt.step([0,N+1],[niveau,niveau],"r")

plt.show()


## Simu version fonction
def simu(tP,tM):
    
    TempsDepart = time.time()
    
    
#     N = 100 # Nombre de sauts par points de la chaîne
    NbrJump = n * N
    
    # Taille des sauts pour toutes les chaînes 
    JumpSizeAbs = npr.choice(val,size=N*n,p=prob)
    
    # Signe des sauts --- /!\ Sous theta /!\
    randTransition=npr.rand(n,N)
    JumpSign = np.apply_along_axis(generateSign, axis=1, arr=randTransition, tP=tP, tM=tM)
    
    # Calcul des Prix min pour les n chaînes mult par Ln
    minP=np.array([(min(P0+np.concatenate([np.arange(1),np.cumsum(JumpSizeAbs[i*N:i*N+N]*JumpSign[i])]))< niveau) * np.product(np.array([Ln(JumpSign[i,j],JumpSign[i,j+1],alpha,tP,tM) for j in np.arange(N-1)])) for i in np.arange(0,n)])
    
    #Affichage de l'estimateur de la proba pour ce niveau
    pEst=np.mean(minP)
    
    
    print("\nDurée d'exécution "+str(time.time()-TempsDepart))
    print ("La proba estimée est " + str(pEst))
    
#     for i in range(5):
#         plt.plot(np.arange(N+1),P0+np.concatenate([np.arange(1),np.cumsum(JumpSizeAbs[i*N:i*N+N]*JumpSign[i])]))
#         plt.plot([0,N+1],[niveau,niveau],"r")
#     plt.show()
    return(pEst)

#Tracé 3D en fonction de thetaP et thetaM
fig = plt.figure()
ax = fig.gca(projection='3d')
x=np.linspace(-0.875,0,5)
y=np.linspace(-0.875,0,5)

X,Y,Z = [],[],[]

for theta1 in x:
    for theta2 in y:
        X.append(theta1)
        Y.append(theta2)
        Z.append(simu(theta1,theta2))

ax.scatter(X,Y,Z)
plt.show()

##
# sEst=pEst*(1-pEst)
# 
# qInf=sps.norm.ppf((1-confiance)/2)
# qSup=sps.norm.ppf((1+confiance)/2)
# 
# bInf=pEst-qSup*sEst/np.sqrt(n)
# bSup=pEst-qInf*sEst/np.sqrt(n)

##
# q1=1e-4
# q2=1-q1
# 
# print("MC estimation quartile au niveau {} avec {} simulations".format(q1, n))
# 
# TempsDepart = time.time()
# 
# # Nombre de sauts sur [0,T], par points de la chaîne
# NbrJump = npr.poisson(T*la,n)
# 
# # Découpage des chaînes
# interval = np.cumsum(np.concatenate([np.arange(1), NbrJump]))
# 
# # Signe des sauts
# randTransition=npr.rand(np.sum(NbrJump))
# JumpSign = [generateSign(randTransition[interval[i]:interval[i+1]],alpha) for i in np.arange(n)]
# 
# Pt=P0+np.array([np.sum(npr.choice(val,size=NbrJump[i],p=prob)*JumpSign[i]) for i in np.arange(n)])
# distrib=np.sort(Pt)
# 
# print("\nDurée d'exécution "+str(time.time()-TempsDepart))
# print(distrib[int(q1*n)],distrib[int(q2*n)])