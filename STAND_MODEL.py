# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.optimize
from scipy.optimize import leastsq
from scipy.optimize import basinhopping
import statistics as stats



plt.close("all")

R=8.3144621
Rerg=R*10**7
Av=6.02*10**23

#Input 
inputdata = {}
inputdata['Aggregation Number'] = float(input("N:"))
inputdata['Temperature'] = float(input("T (Kelvin):"))
inputdata['Surface tension of pure solvent:'] = float(input("TS0 (mN/m):"))
jsondata = json.dumps(inputdata)

Tem=inputdata['Temperature'] 
TS0=inputdata['Surface tension of pure solvent:']
N=inputdata['Aggregation Number']


RT=R*Tem

f=open("octil.txt","r").readlines()

List = [line.strip().split(' ') for line in f]
List = np.array( [line.strip().split(' ') for line in f])
conc=np.array([float(List[i][0]) for i in range(0,len(List))] )
TS_exp=np.array([float(List[i][1]) for i in range(0,len(List))])
u_conc=np.array([float(List[i][2]) for i in range(0,len(List))])
u_TS=np.array([float(List[i][3]) for i in range(0,len(List))])
nn=len(conc)


# =============================================================================
#Equations of the model
# =============================================================================



#Simulation
#Totalc=np.linspace(0.00001*cmc,3*cmc,100) #Total concentration of surfactant


class STAND:
     
    def SEoSLang(F,b,A):
        return TS0-((10**16/Av)*Tem*(R*10**7)*1/A)*np.log(b*F+1) #Langmuir's Equation of State
    
    def Langiso(F,b): #Langmuir's Isotherm
        return b*F/(b*F+1)  
    
    def Micelleconc(T,F): #Micelle concentration
        return (T-F)  
    
    def Balance(NA,GE,T): #Volumetric balance equation
        def FS(F):
            d=NA*np.log(F)-(GE*NA/RT)-np.log(T-F)+np.log(NA)
            return d   
        g=FS
        return g

    def FreeConc(NA,GE,conc):  #Free surfactant concentration
        F=[]
        y=1E-9
        for i in conc:
            l=y
            q=STAND.Balance(NA,GE,i) 
            y=scipy.optimize.bisect(q, l, i)#  at total i concentration of surfactant, y<Free surfactant at i <i, y is the Free surfactant at i-1 point
            F.append(y)
        F=np.array(F)
        return F
    
    def Adjust(Param,conc): 
        
        F=STAND.FreeConc(Param[3],Param[0], conc)  
        M=STAND.Micelleconc(conc,F)
        TS=STAND.SEoSLang(F,Param[1],Param[2]) 
        Theta=STAND.Langiso(F,Param[1])
        
 
        return F,M,TS,Theta

    def Residual(Param,conc,ST_exp,unc): # Objective function
#        print(Param)
        F=STAND.FreeConc(Param[3],Param[0], conc)
        TS=STAND.SEoSLang(F,Param[1],Param[2])
#        s=0
        X=((TS-ST_exp)/unc)**2
#        for i in X:
#            s=s+i
        return np.sqrt(X)
    
    def Aux(Ps_i):   # Auxiliary function for get the initial seeds
        NAg=float(Ps_i[3])
        ELG=float(Ps_i[0])
        beta=float(Ps_i[1])
        Amin=float(Ps_i[2])
#        print(Ps_i)
        F=STAND.FreeConc(NAg,ELG, conc)
        TS=STAND.SEoSLang(F,beta,Amin)
        s=0
        X=((TS-TS_exp)/u_TS)**2
        for i in X:
            s=s+i
        return np.sqrt(s)
    
    def seeds(conc,TS_exp):
        
        def Residual(par,con, TS):
            model=par[0]*con+par[1]
            return ((TS-model))**2
        
        conc2=[conc[i] for i in range(0,7)]
        ln=np.array([np.log(i) for i in conc2])
        TS_exp2=np.array([TS_exp[i] for i in range(0,len(conc2))])
        
        vars=[12,45]
        out1 = leastsq(Residual, vars, args=(ln, TS_exp2))
        c1=np.array(out1)
        d1=c1[0]
        
        
        conc3=[conc[i] for i in range(len(conc)-4,len(conc))]
        ln3=np.array([np.log(i) for i in conc3])
        TS_exp3=np.array([TS_exp[i] for i in range(len(conc)-4,len(conc))])
        
        out2 = leastsq(Residual, vars, args=(ln3, TS_exp3))
        c2=np.array(out2)
        d2=c2[0]
        logcmc=(d2[1]-d1[1])/(d1[0]-d2[0])
        dG=RT*np.log(np.exp((logcmc)))
        
        pmax=TS0-TS_exp[-1]
        bet=np.exp((pmax/-d1[0])-logcmc)
        Amin=(-d1[0]/(298.15*Rerg))**-1*10**16/Av
        return dG,bet,Amin


#Curve fitting 

dG,bet,Amin=STAND.seeds(conc,TS_exp)
Ps=np.array([dG,bet,Amin,N]) #input seeds
minimizer_kwargs = {"method": "BFGS"}
init_seeds = basinhopping(STAND.Aux, Ps, minimizer_kwargs=minimizer_kwargs)
seeds= np.array([init_seeds.x[0],init_seeds.x[1],init_seeds.x[2],init_seeds.x[3]]) #basinhopping for get the global minimum

out1,cov1,infodict,mesg,ier =leastsq(STAND.Residual, seeds, args=(conc, TS_exp,u_TS), full_output=1) #least squares
c=np.array(out1)
resd1=infodict['fvec']
n=0
for i in resd1:
    n=n+i
resvar=(n/(len(conc)-4))
covmat1=resvar*cov1
STD=np.array(np.sqrt(np.diag(covmat1)).tolist())

#Jackknife

def f(par,x, y):
            model=par[0]*x+par[1]
            return ((y-model))**2

L=[i for i in conc]
P=[i for i in TS_exp]
S=[i for i in u_TS]
Jk_Gibbs=[]
Jk_Lang=[]
Jk_Area=[]
Jk_Agg=[]

for i in range(0,nn):
    p=[j for j in L]
    q=[j for j in P]
    t=[j for j in S]
    del(q[i])
    del(p[i])
    del(t[i])
    p=np.array(p)
    q=np.array(q)
    t=np.array(t)
    Ps=np.array([dG,bet,Amin,N]) 
    minimizer_kwargs = {"method": "BFGS"}
    init_seeds = basinhopping(STAND.Aux, Ps, minimizer_kwargs=minimizer_kwargs)
    seeds= np.array([init_seeds.x[0],init_seeds.x[1],init_seeds.x[2],init_seeds.x[3]]) 
    out1,cov1,infodict,mesg,ier =leastsq(STAND.Residual, seeds, args=(p, q,t), full_output=1) 
    c=np.array(out1)
    Jk_Gibbs.append(out1[0])
    Jk_Lang.append(out1[1])
    Jk_Area.append(out1[2])
    Jk_Agg.append(out1[3])
    


prom_Jk_Gibbs=round(stats.mean(Jk_Gibbs),2)
var_Jk_Gibbs=stats.stdev(Jk_Gibbs)

prom_Jk_Area=round(stats.mean(Jk_Area),2)
var_Jk_Area=stats.stdev(Jk_Area)

prom_Jk_Lang=round(stats.mean(Jk_Lang),2)
var_Jk_Lang=stats.stdev(Jk_Lang)

prom_Jk_Agg=round(stats.mean(Jk_Agg),2)
var_Jk_Agg=stats.stdev(Jk_Agg)


xi_sq=STAND.Aux(c)
F,M,TS,Theta=STAND.Adjust(c,conc)

#Gibbs adsorption energy
GsRT=(c[2]*Av/10**16)**-1*Rerg*Tem
dGads=-RT*np.log(GsRT*c[1])

#out 
out = {}
out['Free Gibbs Micellization Energy /kJ/mol'] = round(c[0]/1000,2)
out['STD Free Gibbs Micellization Energy /kJ/mol'] = round(STD[0]/1000,2)

out['Minimum Area /A/molecule '] = round(c[2],2)
out['STD Minimum Area /A/molecule '] = round(STD[2],2)

out['Langmuir s constant'] =round( c[1],2)
out['STD Langmuir s constant'] =round( STD[1],2)

out['Aggregation Number'] = round(c[3],2)
out['STD Aggregation Number'] = round(STD[3],2)

out['Free Gibbs Adsorption Energy /kJ/mol'] = round(dGads/1000,2)
out['Xi'] = round(xi_sq,2)
jsonout = json.dumps(out)
print(jsonout)
print(prom_Jk_Gibbs/1000,var_Jk_Gibbs/1000)
print(prom_Jk_Lang,var_Jk_Lang)
print(prom_Jk_Area,var_Jk_Area)
print(prom_Jk_Agg,var_Jk_Agg)
#Graphs

# Surfactant concentration
plt.figure(1)
plt.plot(conc,F,'b' ,label='Free surfactant')
plt.plot(conc,M,'y' ,label=' Surfactant molecules in micelles')
plt.legend(loc="upper left") 
plt.xlabel('Total concentration/M')
plt.ylabel('Concentration/M')

#Surface tension vs concentration plot
plt.figure(2)
plt.plot(conc,TS,'g')
plt.plot(conc,TS_exp,'ro')
plt.legend(loc="upper right") 
plt.xlabel('Free surfactant concentration/M')
plt.ylabel(r'$\sigma /dinacm^{-1}$')

#Surface coverage
plt.figure(3)
plt.plot(F, Theta, 'b')
plt.xlabel('Free surfactant concentration/M')
plt.ylabel(r'$\theta $')





