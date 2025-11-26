#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

def call_price_BS(sigma,S,T,K,r):
    d1 = (np.log(S/K)+(r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    C = S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return max(C,0)


def find_call_vol(C,S,T,K,r,error, max_NR_iterations=20, B_iterations=4):
    
    ex = 0
    iterations = 0
    
    L = 0
    U = 5
    CL = max(0,S-K)
    CU = call_price_BS(U,S,T,K,r)

    #Sanity Check
    if C < CL or C > CU:
        return pd.NA

    #Bisection Iterations
    for i in range(B_iterations):
        M = ( L + U ) / 2
        CS = call_price_BS(M,S,T,K,r)

        if C == CS:
            return M
        if C < CS:
            U = M
        if C > CS:
            L = M

    sigma = ( L + U ) / 2

    #Newton Raphson for specificity
    while np.abs(CS - C) > error and iterations < max_NR_iterations:
        d1 = (np.log(S/K)+(r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
        vega = S*norm.pdf(d1)*np.sqrt(T)

        #recenter attempt if vega becomes too small
        if np.abs(vega) < 1e-012:
            sigma = 1 + ex
            d1 = (np.log(S/K)+(r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
            vega = S*norm.pdf(d1)*np.sqrt(T)
            ex += 1
        
        sigma = min(max(0.00001,sigma - ((CS-C)/(vega))),15)
        CS = call_price_BS(sigma,S,T,K,r)
        iterations += 1
        if ex > 2:
            return pd.NA
    return sigma


def put_price_BS(sigma,S,T,K,r):
    d1 = (np.log(S/K)+(r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    P = -S*norm.cdf(-d1)+K*np.exp(-r*T)*norm.cdf(-d2)
    return P


def find_put_vol(P,S,T,K,r,error, max_NR_iterations=20, B_iterations=4):

    ex = 0
    iterations = 0

    L = 0
    U = 5
    PL = max(0,S-K)
    PU = put_price_BS(U,S,T,K,r)

    #Sanity Check
    if P < PL or P > PU:
        return pd.NA

    #Bisection Iterations
    for i in range(B_iterations):
        M = ( L + U ) / 2
        PS = put_price_BS(M,S,T,K,r)
        
        if P == PS:
            return M
        if P < PS:
            U = M
        if P > PS:
            L = M

    sigma = ( L + U ) / 2

    #Newton Raphson for specificity
    while np.abs(PS - P) > error and iterations < max_NR_iterations:
        d1 = (np.log(S/K)+(r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
        vega = S*norm.pdf(d1)*np.sqrt(T)

        #recenter attempt if vega becomes too small
        if np.abs(vega) < 1e-12:
            sigma = 1 + ex
            d1 = (np.log(S/K)+(r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
            vega = S*norm.pdf(d1)*np.sqrt(T)
            ex += 1
            
        sigma = min(max(0.00001,sigma - ((PS-P)/(vega))), 15)
        PS = put_price_BS(sigma,S,T,K,r)
        iterations += 1
        if ex > 2:
            return pd.NA
    return sigma



# In[2]:


import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

UNDERLYING = "SPX"

t = yf.Ticker("^" + UNDERLYING)


spot = t.history(period="1d")["Close"].iloc[-1]


expirations = t.options

rows = []
asof = pd.Timestamp.now().normalize()

for exp in expirations:
    chain = t.option_chain(exp)          
    for side_name, df in [("call", chain.calls), ("put", chain.puts)]:
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["option_type"] = side_name
        tmp["expiration"]  = pd.to_datetime(exp)
        tmp["asof"]        = asof
        tmp["underlying"]  = UNDERLYING
        tmp["spot_close"]  = spot.round(6)
 
        tmp["tau"] = ((tmp["expiration"] - tmp["asof"]).dt.days / 365).round(6)
        rows.append(tmp)

options_now = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


rf = yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1] / 100.0
options_now["rf_rate"] = rf


cols = ["asof","underlying","spot_close","expiration","tau","option_type",
        "contractSymbol","strike","lastPrice","bid","ask","volume","openInterest",
        "impliedVolatility","rf_rate"]
options_now = options_now.reindex(columns=[c for c in cols if c in options_now.columns])


options_now = options_now[options_now["tau"] > 2/365]
options_now = options_now[options_now["ask"] != 0]



# In[17]:


err = 0.001

def f(row):
    if pd.isna(row["volume"]) or row["volume"] < 1:
        return pd.NA
    elif row["option_type"] == "call":
        sigma = find_call_vol(C=(row["ask"]+row["bid"])/2,S=row["spot_close"],T=row["tau"],K=row["strike"],r=row["rf_rate"],error=err*(row["ask"]+row["bid"])/2)
        return sigma
    elif row["option_type"] == "put":
        sigma = find_put_vol(P=(row["ask"]+row["bid"])/2,S=row["spot_close"],T=row["tau"],K=row["strike"],r=row["rf_rate"],error=err*(row["ask"]+row["bid"])/2)
        return sigma
    else:
        return pd.NA

def g(row):
    if pd.isna(row["CalVol"]):
        return pd.NA
    elif row["option_type"] == "call":
        return call_price_BS(sigma=row["CalVol"],S=row["spot_close"],T=row["tau"],K=row["strike"],r=row["rf_rate"])
    elif row["option_type"] == "put":
        return put_price_BS(sigma=row["CalVol"],S=row["spot_close"],T=row["tau"],K=row["strike"],r=row["rf_rate"])
    else:
        return pd.NA


options_now["CalVol"] = options_now.apply(f,axis=1)
options_now["CalVP"] = options_now.apply(g,axis=1)

options_now = options_now.dropna(subset=["tau", "strike", "CalVol"]).copy()


    


# In[25]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.interpolate import Rbf


df = options_now.copy()
df["CalVol"] = df["CalVol"].astype(float)
mask = df["CalVol"].notna()
X = df.loc[mask, "tau"].to_numpy()
Y = df.loc[mask, "strike"].to_numpy()
Z = df.loc[mask, "CalVol"].to_numpy()


nx, ny = 50, 50
xi = np.linspace(X.min(), X.max(), nx)
yi = np.linspace(Y.min(), Y.max(), ny)
Xi, Yi = np.meshgrid(xi, yi)

# RBF interpolator with smoothing
rbf = Rbf(
    X, Y, Z,
    function="gaussian", 
    epsilon=0.2,              
    smooth=0.01,              
)

Zi = rbf(Xi, Yi)



# In[26]:


fig = plt.figure(figsize=(12, 5))



ax2 = fig.add_subplot(111, projection="3d")
surf = ax2.plot_surface(Xi, Yi, Zi_smooth, linewidth=0, antialiased=False, cmap=cm.viridis)
ax2.set_xlabel("time to expiry (years)")
ax2.set_ylabel("Strike")
ax2.set_zlabel("IV")
ax2.set_title("Implied Volatility " + UNDERLYING)

ax2.view_init(azim=-30)

fig.colorbar(surf, ax=ax2, shrink=0.70, pad=0.08, label="IV")

plt.tight_layout()
plt.show()






