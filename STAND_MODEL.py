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

RT=R*inputdata['Temperature']
G=(10**16/NA)*inputdata['Temperature']*(R*10**7)*1/inputdata['Minimum Area ']
cmc=np.exp(inputdata['Free Gibbs Micellization Energy']/RT)
TS0=inputdata['Surface tension of pure solvent:']
N=inputdata['Aggregation Number']
Gibbs=inputdata['Free Gibbs Micellization Energy']
Lang=inputdata['Langmuir s constant']

# =============================================================================
#Equations of the model
# =============================================================================

Totalc=np.linspace(0.00001*cmc,3*cmc,100) #Total concentration of surfactant

#Simulation

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
class STAND:
     
    def SEoSLang(F,b,G):
        return TS0-G*np.log(b*F+1)
    def Langiso(F,b):
        return b*F/(b*F+1)  
    def lnK(G,N):
        return -G*N/RT
    def Micelleconc(T,F,N):
        return (T-F)/N   
    def Balance(N, G, RT, T,F): 
        return N*np.log(F)-(G*N/RT)-np.log(T-F)+np.log(N)
    def plotST(F,TS):
        return plt.plot(F,TS,'b')  
    def plotTheta(F,Thet):
        return plt.plot(F,Thet,'b')
#Free surfactant concentration 
#Totalc=conc
F=[]
y=0
for i in Totalc:
    l=y
    p=i    
    func=lambda F: N*np.log(F)-(Gibbs*N)/RT-np.log(i-F)+np.log(N)
    y=scipy.optimize.bisect(func, l, p)
    F.append(y)
F=np.array(F)
# GET Micelle and Monomer concentration      
M=STAND.Micelleconc(Totalc,F,N)
#Surface tension  (Langmuirs EoS and isotherm Free surfactant)
TS=STAND.SEoSLang(F,Lang,G)
Theta=STAND.Langiso(F,Lang)
#logK=STAND.lnK(Gibbs,N)

#Objective function
#def OF(Teor,Exp,Uncer,N,Gibbs,Amin,Lang
#       ):
#    lambda F: N*np.log(F)-(Gibbs*N)/RT-np.log(i-F)+np.log(N)
#    return np.sqrt((Teor-Exp)**2/Uncer**2)
#    
#Val=np.array([Gibbs,N,Lang,G])    
#FOev=OF(TS,TS_exp,u_TS)


plt.figure(1)
plt.plot(Totalc,F,'b' ,label='Free surfactant concentration')
plt.plot(Totalc,M,'y' ,label='Micelle concentration')
plt.legend(loc="center right") 
plt.xlabel('Total concentration/M')
plt.ylabel('Concentracion/M')


plt.figure(2)
STAND.plotST(Totalc,TS)
plt.plot(conc,TS_exp,'ro')
plt.xlabel('Free surfactant concentration/M')
plt.ylabel(r'$\sigma /dinacm^{-1}$')

plt.figure(3)
plt.plot(F, Theta, 'b')
plt.xlabel('Free surfactant concentration/M')
plt.ylabel(r'$\theta $')

