#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 18:57:07 2020

@author: ziranxu
"""

# Through the investment assets to find the efficient frontier  & CML 

import pandas as pd
import numpy as np
import scipy.optimize as sco
import scipy.interpolate as sci
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/ziranxu/Downloads/stocks.csv",index_col=0)
symblos = data.columns
num = len(symblos)
data = data.sort_values(by = ["Date"],ascending=True)

# get the returns of every stock
data["AAPL_r"] = (data["AAPL"]-data["AAPL"].shift(-1))/data["AAPL"].shift(-1)
data["MSFT_r"] = (data["MSFT"]-data["MSFT"].shift(-1))/data["MSFT"].shift(-1)
data["DAL_r"] = (data["DAL"]-data["DAL"].shift(-1))/data["DAL"].shift(-1)
data["TSLA_r"] = (data["TSLA"]-data["TSLA"].shift(-1))/data["TSLA"].shift(-1)
data = data.dropna()
del data["AAPL"],data["MSFT"],data["DAL"],data["TSLA"]
rets  = data/100

# simulate 10000 times, to generate all possible investment combination
prets = []
pvols = []
I = 10000
rets_mean_year  = rets.mean()*252
rets_cov_year = rets.cov()*252

for p in range(I):
    weights = np.random.random(num)
    weights /= np.sum(weights)
    prets.append(np.sum(weights*rets_mean_year))
    pvols.append(np.sqrt(np.dot(weights.T,np.dot(weights,rets_cov_year))))

prets = np.array(prets)
pvols = np.array(pvols)

# get the pics of all portfolios
plt.figure(figsize=(8,5))
plt.scatter(pvols,prets,c = prets/pvols,marker="o")
plt.grid(True)
plt.xlabel("standard deviation")
plt.ylabel("return")
plt.colorbar(label="SR")

# from the pic can tell that these stocks are high related.

#for finding best portfolio
def func_stats(weights,rets):
    weights = np.array(weights)
    prets = np.sum(rets.mean()*weights)*252
    # portfolio's return
    pvols = np.sqrt(np.dot(weights.T,np.dot(weights,rets.cov()*252)))
    #portfolio's volatility 
    return np.array([prets,pvols])
def func_min_vol(weights,rets):
    return func_stats(weights,rets)[1]

# get the efficient frontier
ret_max = max(prets)
ind_min_vol = np.argmin(pvols)
ret_start = prets[ind_min_vol]

trets = np.linspace(ret_start,ret_max,100)
bnds = tuple((0,1) for x in range(num))
tvols = []
for tret in trets:
    cons = ({"type":"eq","fun":lambda x :func_stats(x,rets)[0]-tret},{"type":"eq","fun":lambda x : 1-np.sum(x)})
    res = sco.minimize(lambda x : func_min_vol(x, rets),num*[1/num],method ="SLSQP",bounds = bnds,constraints=cons)
    tvols.append(res["fun"])

tvols = np.array(tvols)

plt.figure(figsize=(8,4))
plt.scatter(tvols,trets,c = trets/tvols,maker = "x")
plt.grid(True)
plt.xlabel("volatility")
plt.ylabel("return")
plt.colorbar(label ="SR")
plt.title("Efficient Frontier")

#CML
ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]
tck = sci.splrep(evols,erets)

def func_ef(x,tck):
    return sci.splev(x,tck,der=0)
def func_def(x,tck):
    return sci.splev(x,tck,der = 1)

def func_equation(p,tck,rf = 0.05):
    eq1 = rf - p[0]
    eq2 = rf+p[1]*p[2]-func_ef(p[2],tck)
    eq3 = p[1]-func_def(p[2],tck)
    return eq1,eq2,eq3

opt =sco.fsolve(lambda x:func_equation(x,tck),[0.05,0.2,0.2])

plt.figure(figsize=(8,4))
plt.scatter(pvols,prets,c=(prets-0.01)/pvols,marker = "o")
plt.plot(evols,erets,"g",lw =2)
plt.plot(opt[2],func_ef(opt[2],tck),"r*")
cx = np.linspace(0,max(pvols))
plt.plot(cx,opt[0]+opt[1]*cx,lw =1.5)
plt.grid(True)
plt.xlabel("volatility")
plt.ylabel("return")
