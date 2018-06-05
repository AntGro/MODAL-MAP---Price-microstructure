import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as sps
import math

##
n = 50000
alpha = 0.95
P0 = 35
T = 4*3600
lamb = 1/300  #temps moyen entre deux sauts : 300s

M = [1,3]
P = [[1/2],[1/4,1/6,1/12]]

i = 0

m = M[i]
p = P[i]

val = np.delete(np.arange(-m,m+1),m)
prob = np.concatenate([p[::-1],p])

def f(x, theta):
    return theta*x

f= np.vectorize(f)


## Proba inf(P)<0

def process(k,theta,proba,pI=P0):
    J = npr.choice(val,size=k,p=proba)
    s = np.cumsum(np.concatenate([np.arange(1),J]))
    normalisation = np.exp(-np.sum(f(J,theta))+lamb*T*(np.dot(np.exp(f(val,theta)),prob)-1))
    return (min(s) < -pI)*normalisation

def estimation_proba(theta):
    lambNew = lamb*np.dot(np.exp(f(val,theta)),prob)
    probNew = np.multiply(np.exp(f(val,theta)),prob)
    probNew = probNew/(np.sum(probNew))
    X = npr.poisson(T*lambNew,n)
    return np.mean(np.array([process(k,theta,proba=probNew) for k in X]))

estimation_proba=np.vectorize(estimation_proba)

def plot():
    theta = np.linspace(-1.2,0,50)
    estimation = estimation_proba(theta)
    plt.plot(theta,estimation)
    plt.show()
    
def bestTheta(inf,sup,nbrPoints, M):
    theta=np.linspace(inf,sup,nbrPoints)
    estimations=[]
    bestStd=+math.inf
    bestTh=inf
    for th in theta:
        for k in range(M):
            estimations.append(estimation_proba(th))
        ecartType=np.std(np.array(estimations))
        if (ecartType<bestStd):
            bestStd=ecartTRype
            bestTh=th
    return bestTh

pEst = estimation_proba(-0.8)
sEst = pEst*(1-pEst)

qInf = sps.norm.ppf((1-alpha)/2)
qSup = sps.norm.ppf((1+alpha)/2)

bInf = pEst-qSup*sEst/np.sqrt(n)
bSup = pEst-qInf*sEst/np.sqrt(n)

## Quantile Q2 - MCMC 

# ATTENTION; n simulations à chaque appel de pEst

def findBounds(nbNiv,P0=35, M=int(8e4), display=True, p=0.7, i=0, T=4*3600, la1=1/660, la2=1/110, N=[1,3], P=[[1/2],[1/4,1/6,1/12]]):
    


def process2(BoundSplit, P0=35, M=int(8e4), display=True, p=0.7, i=0, T=4*3600, la1=1/660, la2=1/110, N=[1,3], P=[[1/2],[1/4,1/6,1/12]]):
    m = N[i]
    val = np.delete(np.arange(-m,m+1),m)
    prob = np.concatenate([P[i][::-1],P[i]])
    K = len(BoundSplit)
    
    if(display):
        print("Les seuils successifs envisagés pour le splitting sont \t")
        print(BoundSplit); print("\n")
        print("Lorsque p = "+ str(p) +" et lambda1 = " + str(la1) + ", lambda2 = "+str(la2))   
        plt.clf()
        
    TempsDepart = time.time()
    
    print("\t Chaine 1 sur "+ str(K))

    # Nombre de sauts sur [0,T], par points de la chaîne
    NbrJump1 = npr.poisson(T*la1,M)
    NbrJump2 = npr.poisson(T*la2,M)
    
    # Temps de sauts sur [0,T], pour toutes les chaînes
    interval1 = np.cumsum(np.concatenate([np.arange(1), NbrJump1]))
    
    # Taille des sauts pour toutes les chaînes (1)
    JumpSize1 = npr.choice(val,size=np.sum(NbrJump1),p=prob)
    
    # Calcul des Prix finaux pour les M chaînes
    endPrice=np.array([P0+np.sum(JumpSize1[interval1[i]:interval1[i+1]]) + npr.choice([-1,1])*(1-NbrJump2[i]%2) for i in np.arange(M)])
    
    # Valeur initiale de la chaîne
    argmin = np.argwhere(endPrice<=BoundSplit[0])                 # indice des chaînes potentielles
    index = npr.choice(argmin.reshape(argmin.size))               # choix au hasard d'un indice
    
    JumpSize1 = JumpSize1[interval1[index]:interval1[index+1]]    # Taille de sauts
    
    PathPoissonInit = JumpSize1    # Chaîne initiale
    
    #Affichage de l'estimateur de la proba pour ce niveau
    Frequence = np.concatenate([np.arange(1),np.cumsum(endPrice<=BoundSplit[0])/np.arange(1,M+1)])
    ProbaEnd = Frequence[-1]
    print ("\t Pour le niveau 1, la proba estimée est " + str(ProbaEnd))
    
    if(display):# Visualisation de la consistance de l'estimateur
        plt.figure(1)
        plt.plot(Frequence, label="niveau 1") 
        
    # BOUCLE pour les AUTRES NIVEAUX  
    for n_level in (1+np.arange(K-1)):
        print("\t Chaine "+ str(n_level+1) + " sur " +str(K))
        
        PathPoisson = PathPoissonInit
        
        NbrJump2 = npr.poisson(T*la2,M)
    
        Frequence = np.zeros(M+1,dtype=float)
        RateAccept = np.zeros(M+1,dtype=float)
        endPricePath = np.zeros(M+1,dtype=float)
        endPricePath[0] = endPrice[index]
        
        for n_chain in np.arange(M):        
            # Quels sauts sont conserves dans la première chaîne
            Conserve=np.random.uniform(0,1,size=PathPoisson.size)<=p
            JumpSizeConserve=PathPoisson[Conserve]
            
            # Combien en ajoute-t-on dans la première chaîne
            NbrAjoute = np.random.poisson((1-p)*la1*T)
            
            # Instants et tailles de sauts ajoutés dans la première chaîne
            NewJumpSize = npr.choice(val,size=NbrAjoute,p=prob)
            
            # Processus candidat à être la nouvelle valeur de la chaine
            NewPathPoisson=np.concatenate([JumpSizeConserve,NewJumpSize])
                    
            # Acceptation-rejet de ce candidat
            NewPrice = np.array(P0+np.sum(NewPathPoisson) + npr.choice([-1,1])*(1-NbrJump2[n_chain]%2))
            
            if NewPrice <= BoundSplit[n_level-1]:    # on accepte
                RateAccept[n_chain+1] = RateAccept[n_chain]+1   # update du taux d'acceptation-rejet
                endPricePath[n_chain+1]=NewPrice
                PathPoisson = NewPathPoisson # mise à jour de la chaine
            else:       # on refuse
                endPricePath[n_chain+1] = endPricePath[n_chain]   #   stockage valeur minimale
                RateAccept[n_chain+1] = RateAccept[n_chain] # update du taux d'acceptation-rejet
            
            # Calcul de l'estimateur de la probabilité
            if endPricePath[n_chain+1]<=BoundSplit[n_level]:
                Frequence[n_chain+1] = Frequence[n_chain]+1
                PathPoissonInit = PathPoisson
            else:
                Frequence[n_chain+1] = Frequence[n_chain]
        
        NewProbaEnd = Frequence[-1]/M
        ProbaEnd = ProbaEnd*NewProbaEnd
        print ("\t Pour le niveau " + str(n_level+1) +", la proba estimée est " + str(NewProbaEnd))   
        
        if(display):
            # Visualisation de la consistance de l'estimateur
            plt.figure(1)
            plt.step(np.arange(Frequence.size),Frequence/(np.arange(M+1,dtype=float)+1), label="niveau %1.0f" %(n_level+1))  
                
            # Visualisation de l'évolution du taux d'acceptation-rejet
            plt.figure(2)
            plt.step(np.arange(RateAccept.size),RateAccept/(np.arange(M+1,dtype=float)+1), label="niveau %1.0f" %(n_level+1)) 
    
    TempsFin = time.time()
    
    print("\n Durée d'exécution "+str(TempsFin-TempsDepart))
    
    if(display):
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
    print("\n La proba que le prix final soit inférieur ou égal à {} en partant de {} est {}\n".format(BoundSplit[-1],P0,ProbaEnd))
    
    return ProbaEnd

def estimation_proba2(theta,c):
    lambNew = lamb*np.dot(np.exp(f(val,theta)),prob)
    probNew = np.multiply(np.exp(f(val,theta)),prob)
    probNew = probNew/(np.sum(probNew))
    X = npr.poisson(T*lambNew,n)
    return np.mean(np.array([process2(k,theta,probNew,c) for k in X]))

def dichotomie(pEst,seuil, sup, theta):
    pas=1
    inf=sup-1
    
    while (pEst(theta,inf)>seuil):
        pas=2*pas
        inf=inf-pas
    
    while(inf<sup):
        c = (sup+inf)//2
        if (pEst(theta,c)<seuil):
            inf = c
        else:
            sup = c
    return inf

#theta = np.linspace(-1.2,0,50)
#dich=np.array([dichotomie(estimation_proba2,0.01,5,th) for th in theta])
#plt.plot(theta,dich)
#plt.show()

## Quantile 2

def process3(seuil,theta):
    res = []
    sIS = []
    
    lambNew = lamb*np.dot(np.exp(f(val,theta)),prob)
    probNew = np.multiply(np.exp(f(val,theta)),prob)
    probNew = probNew/(np.sum(probNew))
    X = npr.poisson(T*lambNew,n)
    
    for i in range(n):
        J = npr.choice(val,size=X[i],p=probNew)
        s = P0+np.sum(J)
        normalisation = np.exp(-np.sum(f(J,theta))-lamb*T*(np.dot(np.exp(f(val,theta)),prob)-1))
        res.append(s)
        sIS.append(normalisation)
    res=np.array(res)
    sIS=np.array(sIS)
    indexSort=np.argsort(res)
    sIsSort=sIS[indexSort]
    #print(sIsSort)
    Y=np.cumsum(sIsSort)
    index=np.argwhere(Y>=n*seuil)[0][0]
    print(index)

    return res[indexSort[index]]

theta=np.linspace(-0.25,-0.05,15)
y=[process3(0.0000001,th) for th in theta]
plt.plot(theta,y)
plt.show()

