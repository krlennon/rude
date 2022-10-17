import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import csv

def maxwell_lr(w, p):
    """ Maxwell linear response function """
    G, tau = p
    Gp = G*(w*tau)**2/(1 + (w*tau)**2)
    Gpp = G*(w*tau)/(1 + (w*tau)**2)
    return Gp, Gpp

def objective(p, w, Gp_data, Gpp_data):
    """ Fitting objective """
    Gp_pred, Gpp_pred = maxwell_lr(w, p)
    res = np.sum(((Gp_pred) - (Gp_data))**2) + np.sum(((Gpp_pred) - (Gpp_data))**2)
    return res

# Get data
data = []
with open("data/gel_saos_1.csv") as file:
    reader = csv.reader(file)
    for row in reader:
        data += [[float(el) for el in row]]

# Prune data
data = np.array(data)
Gp = data[2:,0]
Gpp = data[2:,1]
w = data[2:,3]

# Fit model
obj = lambda p: objective(p, w, Gp, Gpp)
res = minimize(obj, [38000, 1/1.6], bounds=[(0,None),(0,None)])
popt = res.x
print(popt)

# Plot data
fig, ax = plt.subplots(1,1)
ax.loglog(w,Gp,'ro')
ax.loglog(w,Gpp,'bo')

# Make predictions with optimal parameters
xlim = ax.get_xlim()
ylim = ax.get_ylim()
wv = np.logspace(np.log(xlim[0]), np.log(xlim[1]))
Gp_pred, Gpp_pred = maxwell_lr(wv, popt)

# Plot predictions
ax.loglog(wv,Gp_pred,'r--')
ax.loglog(wv,Gpp_pred,'b--')
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.tick_params(which="both",direction="in",right=True,top=True)

plt.show()

