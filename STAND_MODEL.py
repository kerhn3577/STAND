# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import json

plt.close("all")

R=8.3144621
Rerg=R*10**7
NA=6.02*10**23

#Input 
inputdata = {}
inputdata['Free Gibbs Micellization Energy'] = float(input("Gibbs (J/mol):"))
inputdata['Minimum Area '] = float(input("Amin (in Arms per molecule):"))
inputdata['Langmuir s constant'] = float(input("beta:"))
inputdata['Aggregation Number'] = float(input("N:"))
inputdata['Temperature'] = float(input("T (Kelvin):"))
inputdata['Surface tension of pure solvent:'] = float(input("TS0 (mN/m):"))
jsondata = json.dumps(inputdata)

Tem=inputdata['Temperature'] 
TS0=inputdata['Surface tension of pure solvent:']
N=inputdata['Aggregation Number']
Gibbs=inputdata['Free Gibbs Micellization Energy']
Lang=inputdata['Langmuir s constant']
MinA=inputdata['Minimum Area '] 



RT=R*Tem



# =============================================================================
#Equations of the model
# =============================================================================


f=open("octil.txt","r").readlines()
conc=[] 
TS_exp=[]
u_conc=[]
u_TS=[]
List = [line.strip().split(' ') for line in f]
for i in range(0,len(List)):
    conc.append(float(List[i][0]))
    TS_exp.append(float(List[i][1]))
    u_conc.append(float(List[i][2]))
    u_TS.append(float(List[i][3]))
TS_exp=np.array(TS_exp)
u_conc=np.array(u_conc)
u_TS=np.array(u_TS)

#Simulation
#Totalc=np.linspace(0.00001*cmc,3*cmc,100) #Total concentration of surfactant



class STAND:
     
    def SEoSLang(F,b,G):
        return TS0-G*np.log(b*F+1) #Langmuir's Equation of State
    
    def Langiso(F,b): #Langmuir's Isotherm
        return b*F/(b*F+1)  
    
    def Micelleconc(T,F,N): #Micelle concentration
        return (T-F)/N   
    
    def Balance(NA,GE,T):    # Balance equation
        def FS(F):
            return NA*np.log(F)-(GE*NA/RT)-np.log(T-F)+np.log(NA)
        return FS

    def FreeConc(NA,GE,x):  #Free surfactant concentration
        F=[]
        y=0
        for i in x:
            l=y
            p=i
            q=STAND.Balance(NA,GE,i) 
            y=scipy.optimize.bisect(q, l, p)
            F.append(y)
        F=np.array(F)
        return F
    
    def OF(Teor,Exp,Uncer): #Objective function
        s=0
        Xi=((Teor-Exp)/Uncer)**2
        for i in Xi:
            s=s+i
        return  np.sqrt(s)/(len(Exp)-4)
    
    def Adjust(Param,conc):
        
        G=(10**16/NA)*Tem*(R*10**7)*1/Param[3]
        F=np.array(STAND.FreeConc(Param[1],Param[0], conc) )      
        M=np.array(STAND.Micelleconc(conc,F,Param[1])) #Micelle concentration
        TS=np.array(STAND.SEoSLang(F,Param[2],G)) #Surface tension  (Langmuirs EoS and isotherm)
        Theta=np.array(STAND.Langiso(F,Param[2])  ) 
        Xi_sq=STAND.OF(TS,TS_exp,u_TS)
 
        return F,M,TS,Theta,Xi_sq

#Optimization        
Ps=np.array([Gibbs,N,Lang,MinA])    
#cmc=np.exp(Ps[0]/RT)

F,M,TS,Theta,Xi_sq=STAND.Adjust(Ps,conc)

print(Xi_sq)


#Graphs
plt.figure(1)
plt.plot(conc,F,'b' ,label='Free surfactant concentration')
plt.plot(conc,M,'y' ,label='Micelle concentration')
plt.legend(loc="center right") 
plt.xlabel('Total concentration/M')
plt.ylabel('Concentracion/M')


plt.figure(2)
plt.plot(conc,TS)
plt.plot(conc,TS_exp,'ro')
plt.xlabel('Free surfactant concentration/M')
plt.ylabel(r'$\sigma /dinacm^{-1}$')

plt.figure(3)
plt.plot(F, Theta, 'b')
plt.xlabel('Free surfactant concentration/M')
plt.ylabel(r'$\theta $')

