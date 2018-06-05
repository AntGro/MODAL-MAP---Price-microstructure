import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import time

###############################################################################
## Q1 Splitting + MCMC --- Fonction
###############################################################################

def Q1_MCMC(P0=35, niveau=0, M=int(8e4), display=True, n=int(1e3), ratio=0.1, p=0.7, i=0, T=4*3600, la=1/300, N=[1,3], P=[[1/2],[1/4,1/6,1/12]]):
    BoundSplit = Q1_MCMC_Seuil(P0, niveau, n, ratio, p, i, T, la, N, P)
    return Q1_MCMC_Body(BoundSplit, P0, M, display, p, i, T, la, N, P)

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


def Q1_MCMC_Seuil(P0=35, niveau=0, n=int(1e3), ratio=0.1, p=0.7, i=0, T=4*3600, la=1/300, N=[1,3], P=[[1/2],[1/4,1/6,1/12]]):
    m = N[i]
    val = np.delete(np.arange(-m,m+1),m)
    prob = np.concatenate([P[i][::-1],P[i]])
    # Nombre de sauts sur [0,T], par points de la chaîne
    NbrJump=npr.poisson(T*la,n)
    
    # Temps de sauts sur [0,T], pour toutes les chaînes
    TimeJump = T*np.random.uniform(0,1,size=np.sum(NbrJump))
    
    # Taille des sauts pour toutes les chaînes
    JumpSize = npr.choice(val,size=np.sum(NbrJump),p=prob)
    
    # Calcul des prix min pour chaque chaîne
    interval = np.cumsum(np.concatenate([np.arange(1),NbrJump]))
    minP = np.array([min(P0+np.concatenate([np.arange(1),np.cumsum(JumpSize[interval[i]:interval[i+1]])])) for i in np.arange(n)])
    
    #previousSeuil (on veut que les seuils soient strictement décroissants)
    previousSeuil = P0
    # Premier seuil a1
    a = [np.sort(minP)[int(ratio*n)]]
    if(a[-1] == previousSeuil):
        a[-1] -= 1
        
    # Valeur initiale de la chaîne
    argmin = np.argwhere(minP<=a[-1])                               # indice des chaînes potentielles
    index = npr.choice(argmin.reshape(argmin.size))                 # choix au hasard d'un indice
    TimeJump = np.sort(TimeJump[interval[index]:interval[index+1]]) # Temps de sauts 
    JumpSize = JumpSize[interval[index]:interval[index+1]]          # Taille de sauts
    
    PathPoissonInit = [TimeJump,JumpSize]                           # Chaîne initiale
    step = 1
    
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
            NbrAjoute = npr.poisson((1-p)*la*T)
            
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
        previousSeuil = a[-1]
        a.append(np.sort(minP)[int(ratio*n)])
        if(a[-1] == previousSeuil and min(minP) < a[-1]):
            a[-1] -= 1
        elif(a[-1] == previousSeuil):
            del a[-1]
            n = int(1.5*n) #on doit faire plus de simulations pour obtenir un nouveau seuilpreviousSeuil = a[-1]
        
        print("Seuil à l'étape {} : {}".format(step,a[-1]))
        
        # Valeur initiale de la chaîne
        argmin=np.argwhere(minP<=a[-1]) # indice des chaînes potentielles
        index=npr.choice(argmin.reshape(argmin.size)) # choix au hasard d'un indice
        TimeJump=TimeJumps[index] # Temps de sauts 
        JumpSize=JumpSizes[index] # Taille de sauts
    
        PathPoissonInit=[TimeJump,JumpSize] # Chaîne initiale
        step += 1
        
    a[-1]=niveau-1
    
    print("Valeur des seuils : {}".format(a))
    return a 
#
#
#    
def Q1_MCMC_Body(BoundSplit, P0=35, M=int(8e4), display=True, p=0.7, i=0, T=4*3600, la=1/300, N=[1,3], P=[[1/2],[1/4,1/6,1/12]]):
    m = N[i]
    val = np.delete(np.arange(-m,m+1),m)
    prob = np.concatenate([P[i][::-1],P[i]])
    K = len(BoundSplit)
    
    if(display):
        print("Les seuils successifs envisagés pour le splitting sont \t")
        print(BoundSplit); print("\n")
        print("Lorsque p = "+ str(p) +" et lambda = " + str(la))   

    
    TempsDepart = time.time()
    
    plt.clf()

    print("\t Chaine 1 sur "+ str(K))

    # Nombre de sauts sur [0,T], par points de la chaîne
    NbrJump = npr.poisson(T*la,M)
    
    # Taille des sauts pour toutes les chaînes (1)
    JumpSize = npr.choice(val,size=np.sum(NbrJump),p=prob)
    
    # Temps de sauts sur [0,T], pour toutes les chaînes
    interval = np.cumsum(np.concatenate([np.arange(1), NbrJump]))
    
    TimeJump = T*np.random.uniform(0,1,size=np.sum(NbrJump))
    TimeJump = [np.sort(TimeJump[interval[i]:interval[i+1]]) for i in np.arange(M)]
    
    # Calcul des Prix min pour les M chaînes
    minP=np.array([min(P0+np.concatenate([np.arange(1),np.cumsum(JumpSize[interval[i]:interval[i+1]])])) for i in np.arange(M)])
    
    # Valeur initiale de la chaîne
    argmin = np.argwhere(minP<=BoundSplit[0])                     # indice des chaînes potentielles
    index = npr.choice(argmin.reshape(argmin.size))               # choix au hasard d'un indice
    TimeJump = TimeJump[index]                                    # Temps de sauts (1)
    JumpSize = JumpSize[interval[index]:interval[index+1]]        # Taille de sauts
    
    PathPoissonInit=[TimeJump,JumpSize]                           # Chaîne initiale
    
    #Affichage de l'estimateur de la proba pour ce niveau
    Frequence = np.concatenate([np.arange(1),np.cumsum(minP<=BoundSplit[0])/np.arange(1,M+1)])
    ProbaEnd = Frequence[-1]
    print ("\t Pour le niveau 1, la proba estimée est " + str(ProbaEnd))
    
    if(display):# Visualisation de la consistance de l'estimateur
        plt.figure(1)
        plt.step(np.arange(Frequence.size),Frequence, label="niveau 1")
        
    # BOUCLE pour les AUTRES NIVEAUX  
    for n_level in (1+np.arange(K-1)):
        print("\t Chaine "+ str(n_level+1) + " sur " +str(K))
        
        PathPoisson = PathPoissonInit
        
        Frequence = np.zeros(M+1,dtype=float)
        RateAccept = np.zeros(M+1,dtype=float)
        MinPath = np.zeros(M+1,dtype=float)
        MinPath[0] = min(P0+np.cumsum(np.concatenate([np.arange(1),PathPoisson[1]])))
        
        for n_chain in np.arange(M):
            # Quels sauts sont conserves
            Conserve=np.random.uniform(0,1,size=PathPoisson[0].size)<=p
            JumpTimeConserve = PathPoisson[0][Conserve]
            JumpSizeConserve = PathPoisson[1][Conserve]
            
            # Combien en ajoute-t-on
            NbrAjoute = np.random.poisson((1-p)*la*T)
            
            # Instants de sauts ajoutés
            NewTimeJump = np.sort(T*np.random.uniform(0,1,NbrAjoute))
            NewJumpSize = npr.choice(val,size=NbrAjoute,p=prob)
            
            # Processus candidat à être la nouvelle valeur de la chaine
            NewPathPoisson = fusion(JumpTimeConserve,JumpSizeConserve,NewTimeJump,NewJumpSize)
            
            # Acceptation-rejet de ce candidat
            NewPrice = min(P0+np.cumsum(np.concatenate([np.arange(1),NewPathPoisson[1]])))
            
            if NewPrice <= BoundSplit[n_level-1]:    # on accepte
                PathPoisson = NewPathPoisson    # mise à jour de la chaine
                MinPath[n_chain+1] = NewPrice    # stockage valeur minimale
                RateAccept[n_chain+1] = RateAccept[n_chain]+1   # update du taux d'acceptation-rejet
            else:       # on refuse
                MinPath[n_chain+1] = MinPath[n_chain]   #   stockage valeur minimale
                RateAccept[n_chain+1] = RateAccept[n_chain] # update du taux d'acceptation-rejet
            
            # Calcul de l'estimateur de la probabilité
            if MinPath[n_chain+1] <= BoundSplit[n_level]:
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
    print("\n La proba de ruine estimée en partant de {} est {}\n".format(P0,ProbaEnd))
    
    return ProbaEnd
    
def Q1_MCMCIC(P0=35, niveau=0, pVecteur=[0.7], NbrAlgo=25, M=int(5e4), display=True, i=0, T=4*3600, la=1/300, N=[1,3], P=[[1/2],[1/4,1/6,1/12]]):
    plt.close()
    
    val=np.delete(np.arange(-N[i],N[i]+1),N[i])
    prob=np.concatenate([P[i][::-1],P[i]])
    
    length_p = len(pVecteur)
    LengthTrajVecteur=[M for i in range(length_p)]

    TempsDepart = time.time()

    StockProbaEnd = np.zeros((length_p,NbrAlgo))
    # Boucle sur les valeurs de p
    for n_p in np.arange(length_p):
        p = pVecteur[n_p]
        BoundSplit = Q1_MCMC_Seuil(P0, 0,int(1e3),0.1,p, i, T, la, N, P)
        LengthTraj = LengthTrajVecteur[n_p]
        print("Lorsque p = "+ str(p) +" et lambda = " + str(la))  
        print("\t BoundSplit : " + str(BoundSplit))  
        plt.clf()
        
        # Boucle sur les differents runs independants
        for n_algo in np.arange(NbrAlgo):
            print("Run "+ str(n_algo+1)+ " sur " + str(NbrAlgo))
            ProbaEnd = Q1_MCMC_Body(BoundSplit, P0, M, False, p, i, T, la, N, P)
            StockProbaEnd[n_p,n_algo] = ProbaEnd
        print("La proba de ruine estimée est " + str(np.mean(StockProbaEnd[n_p,:])))
        print("IC asymptotique a 95% : +/- "+ str(np.std(StockProbaEnd[n_p,:])/np.sqrt(n_algo)))
    
    TempsFin = time.time()
    print("Durée d'exécution "+str(TempsFin-TempsDepart))
    
    if(display) : 
        plt.clf()
        
        plt.figure(1)
        plt.title("Boxplot de " + str(NbrAlgo) +" estimateurs  pour lambda= " +str(la) + " et M= "+str(LengthTrajVecteur[0]))
        plt.boxplot(np.transpose(StockProbaEnd), positions= [2*i+1 for i in np.arange(length_p)], labels = [str(x) for x in pVecteur])
        plt.xlabel("valeur de p")
        plt.ylabel("Valeur de l'estimateur")
        plt.savefig("Boxplot.lambda"+ str(la)+".M"+str(LengthTrajVecteur[0])+".pdf")
        
        plt.figure(2)
        M = np.mean(StockProbaEnd, axis=1)
        S = np.std(StockProbaEnd, axis=1)
        plt.plot(pVecteur, S/M, 'd-')
        plt.xlabel("valeur de p")
        plt.ylabel("ratio ecart type/moyenne")
        plt.title("lambda= " +str(la)+ " et M= "+str(LengthTrajVecteur[0]))
        plt.grid()
        plt.savefig("Ratios.lambda"+ str(la)+".M"+str(LengthTrajVecteur[0])+".pdf")
        
        
        plt.show()
    
    return (StockProbaEnd)