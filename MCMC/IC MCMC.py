# -*- coding: utf-8 -*-
import numpy as np
import  matplotlib.pyplot as plt
import time

############;###################################################################
## Question 9
###############################################################################
plt.close()

P0 = 10
T=4*3600
la = 1/300

N=[1,3]
P=[[1/2],[1/4,1/6,1/12]]

i=0

m=N[i]
p=P[i]

val=np.delete(np.arange(-m,m+1),m)
prob=np.concatenate([p[::-1],p])

pVecteur=[0.7]
pVecteur = [0.1,0.2,0.3,0.4, 0.5, 0.6,0.7,0.8, 0.9]
length_p = len(pVecteur)
LengthTrajVecteur=[int(1e4) for i in range(length_p)]

#la,K = 0.05,8
# pVecteur = [0.1, 0.5, 0.9]
# length_p = len(pVecteur)
# LengthTrajVecteur=[int(1e4) for i in np.arange(length_p)]


# Nombre de réalisations indépendantes de l'estimateur
NbrAlgo = 10


# calcul des seuils successifs
BoundSplit = P0*(1-((np.arange(K,dtype=float)+1)**(1/2)/K**(1/2)))-1
print("les seuils successifs envisagés pour le splitting sont \t")
print(BoundSplit)


TempsDepart = time.time()

StockProbaEnd = np.zeros((length_p,NbrAlgo))
# Boucle sur les valeurs de p
for n_p in np.arange(length_p):
     p = pVecteur[n_p]
     LengthTraj = LengthTrajVecteur[n_p]
     print("Lorsque p = "+ str(p) +" et lambda = " + str(la))   
     plt.clf()
     
     # Boucle sur les differents runs independants
     for n_algo in np.arange(NbrAlgo):
        print("Run "+ str(n_algo+1)+ " sur " + str(NbrAlgo))
        
        ##----------
        # NIVEAU 1
        ##----------
        # Nombre de sauts sur [0,T], par points de la chaîne
        NbrJump = npr.poisson(lam=la*T,size=(1,LengthTraj))
        Frequence = np.zeros(LengthTraj+1,dtype=float)
        #Boucle pour simuler la chaine de Markov
        print("\t Chaine 1 sur "+ str(K))
        
        for n_chain in np.arange(LengthTraj):
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
        ProbaEnd = Frequence[-1]/LengthTraj
        #print ("\t Pour le niveau 1, la proba estimée est " + str(ProbaEnd))
        # Visualisation de la consistance de l'estimateur
        plt.figure(1)
        plt.plot(Frequence/(np.arange(LengthTraj+1)+1)) 
            
        ## BOUCLE pour les AUTRES NIVEAUX
        for n_level in (1+np.arange(K-1)):
            print("\t Chaine "+ str(n_level+1) + " sur " +str(K))
            Frequence = np.zeros(LengthTraj+1,dtype=float)
            RateAccept = np.zeros(LengthTraj+1,dtype=float)
            PathPoisson = PathPoissonInit
            MinPath = np.zeros(LengthTraj+1,dtype=float)
            MinPath[0] = min(PathPoisson[2])
            for n_chain in np.arange(LengthTraj):
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
            NewProbaEnd = Frequence[-1]/LengthTraj
            ProbaEnd = ProbaEnd*NewProbaEnd
            #print ("\t Pour le niveau " + str(n_level) +", la proba estimée est " + str(NewProbaEnd))    
            
            #Visulation de la consistance et de l'évoution du tax d'acceptation-rejet pour un run
            #if n_algo==1:
                ## Visualisation de la consistance de l'estimateur
                #plt.figure(1)
                #plt.plot(Frequence/(np.arange(LengthTraj+1)+1),label="niveau "+ str(n_level))  
                # Visualisation de l'évolution du taux d'acceptation-rejet
                #plt.figure(2)
                #plt.plot(RateAccept/(np.arange(LengthTraj+1)+1),label="niveau "+ str(n_level)) 
                    
        #plt.figure(1)
        #plt.title("Consistance des estimateurs des probabilites pour p= "+ str(p))
        #plt.legend(loc="best")
        #plt.grid()
        #plt.figure(2)
        #plt.title("Evolution du taux d'acceptation-rejet pour p= "+ str(p))
        #plt.legend(loc="best")
        #plt.grid()
    
        # Calcul de l'estimateur 
        #print("Run "+ str(n_algo+1)+": la proba de ruine estimée est " + str(ProbaEnd))
        StockProbaEnd[n_p,n_algo] = ProbaEnd
     print("La proba de ruine estimée est " + str(np.mean(StockProbaEnd[n_p,:])))
     print("IC asymptotique a 95% : +/- "+ str(np.std(StockProbaEnd[n_p,:])/np.sqrt(n_algo)))

TempsFin = time.time()
print("Durée d'exécution "+str(TempsFin-TempsDepart))

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

