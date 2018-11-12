333# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.optimize
from scipy.optimize import leastsq
from scipy.optimize import basinhopping
#from lmfit import minimize, Parameters

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
     
    def SEoSLang(F,b,A):
        return TS0-((10**16/NA)*Tem*(R*10**7)*1/A)*np.log(b*F+1) #Langmuir's Equation of State
    
    def Langiso(F,b): #Langmuir's Isotherm
        return b*F/(b*F+1)  
    
    def Micelleconc(T,F): #Micelle concentration
        return (T-F)  
    
    def Balance(NA,GE,T):    # Balance equation
        def FS(F):
            d=NA*np.log(F)-(GE*NA/RT)-np.log(T-F)+np.log(NA)
            return d   
        g=FS
        return g

    def FreeConc(NA,GE,conc):  #Free surfactant concentration
        F=[]
        y=0
        for i in conc:
            p=y
            l=i
            q=STAND.Balance(NA,GE,i) 
            y=scipy.optimize.bisect(q, l, p)
            F.append(y)
        F=np.array(F)
        return F
    
    def Adjust(Param,conc,data,uncer):
        
        F=STAND.FreeConc(Param[3],Param[0], conc)  
        M=STAND.Micelleconc(conc,F)
        TS=STAND.SEoSLang(F,Param[1],Param[2]) 
        Theta=STAND.Langiso(F,Param[2])  
 
        return F,M,TS,Theta

    def Residual(Param,conc,data,uncer):
        print(Param)
        F=STAND.FreeConc(Param[3],Param[0], conc)
        TS=STAND.SEoSLang(F,Param[1],Param[2])
#        s=0
        X=((TS-data)/uncer)**2
#        for i in X:
#            s=s+i
        return np.sqrt(X)

#Curve fitting 

Ps=np.array([Gibbs,Lang,MinA,N])
#cmc=np.exp(Ps[0]/RT)

def func(Ps_i):
    NAg=float(Ps_i[0])
    ELG=float(Ps_i[1])
    beta=float(Ps_i[2])
    Amin=float(Ps_i[3])
    print(Ps_i)
    F=STAND.FreeConc(NAg,ELG, conc)
    TS=STAND.SEoSLang(F,beta,Amin)
    s=0
    X=((TS-TS_exp)/u_TS)**2
    for i in X:
       s=s+i
    return s

minimizer_kwargs = {"method": "BFGS"}
ret = basinhopping(func, Ps, minimizer_kwargs=minimizer_kwargs)

seeds= np.array([ret.x[0],ret.x[1],ret.x[2],ret.x[3]])

out=leastsq(STAND.Residual, seeds, args=(conc, TS_exp, u_TS))


c=np.array(out)
d=c[0]
print(d)
F,M,TS,Theta=STAND.Adjust(d,conc,TS_exp,u_TS)
F1,M1,TS1,Theta1=STAND.Adjust(Ps,conc,TS_exp,u_TS)
print(d)
print(STAND.Residual(d,conc,TS_exp,u_TS))
#Graphs
plt.figure(1)
plt.plot(conc,F,'b' ,label='Free surfactant')
plt.plot(conc,M,'y' ,label=' Surfactant molecules in micelles')
plt.legend(loc="center right") 
plt.xlabel('Total concentration/M')
plt.ylabel('Concentration/M')


plt.figure(2)
plt.plot(conc,TS)
#plt.plot(conc,TS1)
plt.plot(conc,TS_exp,'ro')
plt.xlabel('Free surfactant concentration/M')
plt.ylabel(r'$\sigma /dinacm^{-1}$')

plt.figure(3)
plt.plot(F, Theta, 'b')
plt.xlabel('Free surfactant concentration/M')
plt.ylabel(r'$\theta $')

