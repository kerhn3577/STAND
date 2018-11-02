# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import json
from scipy.optimize import leastsq

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

List = [line.strip().split(' ') for line in f]
List = np.array( [line.strip().split(' ') for line in f])
conc=np.array([float(List[i][0]) for i in range(0,len(List))] )
TS_exp=np.array([float(List[i][1]) for i in range(0,len(List))])
u_conc=np.array([float(List[i][2]) for i in range(0,len(List))])
u_TS=np.array([float(List[i][3]) for i in range(0,len(List))])


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
    
    def Adjust(Param,conc,data,uncer):
        
        G=(10**16/NA)*Tem*(R*10**7)*1/Param[3]
        F=STAND.FreeConc(Param[1],Param[0], conc)  
        M=STAND.Micelleconc(conc,F,Param[1]) #Micelle concentration
        TS=STAND.SEoSLang(F,Param[2],G) #Surface tension  (Langmuirs EoS and isotherm)
        Theta=STAND.Langiso(F,Param[2])  
        Xi_sq=STAND.OF(TS,data,uncer)
 
        return F,M,TS,Theta,Xi_sq

    
def Residual(Param,conc,data,uncer):
        
    G=(10**16/NA)*Tem*(R*10**7)*1/Param[3]
    F=STAND.FreeConc(Param[1],Param[0], conc)  
    TS=STAND.SEoSLang(F,Param[2],G) #Surface tension  (Langmuirs EoS and isotherm)

    return ((TS-data)/uncer)**2


#Curve fitting 
        

#params = Parameters()
#params.add('amp', value=10, vary=False)
#params.add('decay', value=0.007, min=0.0)
#params.add('phase', value=0.2)
#params.add('frequency', value=3.0, max=10)
Ps=np.array([Gibbs,N,Lang,MinA])    
#cmc=np.exp(Ps[0]/RT)
vars = Ps
out = leastsq(Residual, vars, args=(conc, TS_exp, u_TS))
c=np.array(out)
d=c[0]
F,M,TS,Theta,Xi_sq=STAND.Adjust(d,conc,TS_exp,u_TS)


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

