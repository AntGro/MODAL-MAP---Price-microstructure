import numpy as np
import numpy.random as npr
import scipy.stats as sps
import time
import matplotlib.pyplot as plt

###############################################################################
## Q3. Changement de loi
###############################################################################
plt.close()

P0 = 1000
niveau = 0

T = 8*3600*250 # Nombre de sauts par points de la chaîne
la = 1/300

N = [1,3]
P = [[1], [1/2,1/3,1/6]]

i=1

m = N[i]
p = P[i]

val = np.arange(1,m+1)
prob = p

n=int(5e3)

# paramètre de la matrice de transition
alpha = -0.875

#Nouveau paramètre de transition
theta = 0.5



confiance = 0.95    

##----------
# Estimation de la probabilité d'avoir un prix négatif
##----------

 #étant donné un array de nombres aléatoires 'rand' renvoie génère la chaîne de Markov des signes suivant la matrice de transition de paramètre alpha
def generateSign(rand,a=alpha):
    s=[2*(rand[0]<0.5)-1]
    for i in np.arange(1,rand.size):
        s.append((2*(rand[i]<((1+s[-1]*a)/2))-1))
    return np.array(s)

def Ln(x, y, a=alpha, t=theta): # retourne le rapport de vraissemblance de P_a par rapport à P_t
    if (x == y):
        return (1+a)/(1+t)
    return (1-a)/(1-t)



## Simu script
# TempsDepart = time.time()
# 
# # Nombre de sauts sur [0,T], par points de la chaîne
# NbrJump = npr.poisson(T*la,n) 
# 
# # Découpage des chaînes
# interval = np.cumsum(np.concatenate([np.arange(1), NbrJump]))
# 
# # Taille des sauts pour toutes les chaînes 
# JumpSizeAbs = npr.choice(val,size=np.sum(NbrJump),p=prob)
# 
# # Signe des sauts --- /!\ Sous theta /!\
# randTransition=npr.rand(np.sum(NbrJump))
# JumpSign = [generateSign(randTransition[interval[i]:interval[i+1]],theta) for i in np.arange(n)]
# 
# # Calcul des Prix min multipliés par Ln pour chacune des n chaînes 
# minP=np.array([(min(P0+np.concatenate([np.arange(1),np.cumsum(JumpSizeAbs[interval[i]:interval[i+1]]*JumpSign[i])]))< 0) * np.product(np.array([Ln(JumpSign[i][j],JumpSign[i][j+1]) for j in np.arange(len(JumpSign[i])-1)])) for i in np.arange(0,n)])
# 
# #Affichage de l'estimateur de la proba pour ce niveau
# pEst=np.mean(minP)
# 
# 
# print("\nDurée d'exécution "+str(time.time()-TempsDepart))
# print ("La proba estimée est " + str(pEst))
# 
# for i in range(5):
#     plt.step(np.arange(NbrJump[i]+1),P0+np.concatenate([np.arange(1),np.cumsum(JumpSizeAbs[interval[i]:interval[i+1]]*JumpSign[i])]))
#    
# plt.plot([0,max(NbrJump[:5])+1],[0,0],"r")
# plt.show()


## Simu version fonction
def simu(t):
    
    TempsDepart = time.time()
    # Nombre de sauts sur [0,T], par points de la chaîne
    NbrJump = npr.poisson(T*la,n) 
    
    # Découpage des chaînes
    interval = np.cumsum(np.concatenate([np.arange(1), NbrJump]))
    
    # Taille des sauts pour toutes les chaînes 
    JumpSizeAbs = npr.choice(val,size=np.sum(NbrJump),p=prob)
    
    # Signe des sauts --- /!\ Sous theta /!\
    randTransition=npr.rand(np.sum(NbrJump))
    JumpSign = [generateSign(randTransition[interval[i]:interval[i+1]],t) for i in np.arange(n)]
    
    # Calcul des Prix min multipliés par Ln pour chacune des n chaînes 
    minP=np.array([(min(P0+np.concatenate([np.arange(1),np.cumsum(JumpSizeAbs[interval[i]:interval[i+1]]*JumpSign[i])]))< 0) * np.product(np.array([Ln(JumpSign[i][j],JumpSign[i][j+1],alpha,t) for j in np.arange(len(JumpSign[i])-1)])) for i in np.arange(0,n)])
    
    #Affichage de l'estimateur de la proba pour ce niveau
    pEst=np.mean(minP)
    
    
    print("\nDurée d'exécution "+str(time.time()-TempsDepart))
    print ("La proba estimée est " + str(pEst))
    
#     for i in range(5):
#         plt.plot(np.arange(N+1),P0+np.concatenate([np.arange(1),np.cumsum(JumpSizeAbs[i*N:i*N+N]*JumpSign[i])]))
#         plt.plot([0,N+1],[niveau,niveau],"r")
#     plt.show()
    return(pEst)

plt.figure(1)
theta1=np.linspace(-0.875,0.5,100)
y1=[simu(th) for th in theta1]
plt.plot(theta1,y1)

sEst=np.std(minP)

qInf=sps.norm.ppf((1-confiance)/2)
qSup=sps.norm.ppf((1+confiance)/2)

bInf=pEst-qSup*sEst/np.sqrt(n)
bSup=pEst-qInf*sEst/np.sqrt(n)

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
# JumpSign = [generateSign(randTransition[interval[i]:interval[i+1]],theta) for i in np.arange(n)]
# 
# Pt=P0+np.array([np.sum(npr.choice(val,size=NbrJump[i],p=prob)*JumpSign[i]) for i in np.arange(n)])
# distrib=np.sort(Pt)
# 
# print("\nDurée d'exécution "+str(time.time()-TempsDepart))
# print(distrib[int(q1*n)],distrib[int(q2*n)])

##Quantile

def process(seuil,t):
        
    #Nombre de sauts sur [0,T], par points de la chaîne
    NbrJump = npr.poisson(T*la,n) 
    
    # Découpage des chaînes
    interval = np.cumsum(np.concatenate([np.arange(1), NbrJump]))
    
    # Taille des sauts pour toutes les chaînes 
    JumpSizeAbs = npr.choice(val,size=np.sum(NbrJump),p=prob)
    
    # Signe des sauts --- /!\ Sous theta /!\
    randTransition=npr.rand(np.sum(NbrJump))
    JumpSign = [generateSign(randTransition[interval[i]:interval[i+1]],t) for i in np.arange(n)]
    # Calcul des Prix min pour les n chaînes mult par Ln
    res= np.array([(P0+np.sum(JumpSizeAbs[interval[i]:interval[i+1]]*JumpSign[i])) for i in np.arange(0,n)])
    sIS=np.array([np.product(np.array([Ln(JumpSign[i][j],JumpSign[i][j+1],alpha,t) for j in np.arange(len(JumpSign[i])-1)])) for i in np.arange(0,n)])
    
    indexSort=np.argsort(res)
    sIsSort=sIS[indexSort]
    Y=np.cumsum(sIsSort)
    index=np.argwhere(Y>=n*seuil)[0][0]

    return res[indexSort[index]]

plt.figure(2)
theta2=np.linspace(-0.875,0.5,100)
y2=[process(0.01,th) for th in theta2]
plt.plot(theta2,y2)
plt.show()




