import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_mesh(x, y, v, half="top"):
    # Make a rectangular mesh from the coordinates and velocities
    # Prune to half the space
    isTop = (half == "top")

    if isTop:
        x = x[y >= 0]
        v = v[y >= 0]
        y = y[y >= 0]
    else:
        x = x[y < 0]
        v = v[y < 0]
        y = y[y < 0]

    # Make X and Y mesh grids
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    X, Y = np.meshgrid(x_unique,y_unique)

    # Fill in the V grid
    V = np.empty_like(X)
    V[:,:] = np.nan
    for i in range(len(x)):
        m = np.where(y_unique == y[i])[0][0]
        n = np.where(x_unique == x[i])[0][0]
        V[m,n] = v[i]

    return X, Y, V

# Import data for the centerline velocity
df_giesekus_centerline = pd.read_csv("simdata/t10_U_centerline_Giesekus.csv")
x_giesekus = df_giesekus_centerline["Points:0"].to_numpy()
vx_giesekus = df_giesekus_centerline["U:0"].to_numpy()

df_rude_centerline = pd.read_csv("simdata/t10_U_centerline_RUDE.csv")
x_rude = df_rude_centerline["Points:0"].to_numpy()
vx_rude = df_rude_centerline["U:0"].to_numpy()

df_oldroyd_centerline = pd.read_csv("simdata/t10_U_centerline_OldroydB.csv")
x_oldroyd = df_oldroyd_centerline["Points:0"].to_numpy()
vx_oldroyd = df_oldroyd_centerline["U:0"].to_numpy()

# Plot centerline velocities
fig1, ax1 = plt.subplots(1,1)
ax1.plot(x_oldroyd, vx_oldroyd, 'b--', lw=2)
ax1.plot(x_rude, vx_rude, 'r-', lw=2)
ax1.plot(x_giesekus, vx_giesekus, 'k--', lw=2)
ax1.set_xlim([-7.5,5])
ax1.set_ylim([0.2, 1.7])
ax1.tick_params(which="both", direction="in", top=True, right=True)

# Import data for the outlet velocity
df_giesekus_outlet = pd.read_csv("simdata/t10_U_outlet_Giesekus.csv")
y_giesekus = df_giesekus_outlet["Points:1"].to_numpy()
vx_giesekus = df_giesekus_outlet["U:0"].to_numpy()

df_rude_outlet = pd.read_csv("simdata/t10_U_outlet_RUDE.csv")
y_rude = df_rude_outlet["Points:1"].to_numpy()
vx_rude = df_rude_outlet["U:0"].to_numpy()

df_oldroyd_outlet = pd.read_csv("simdata/t10_U_outlet_OldroydB.csv")
y_oldroyd = df_oldroyd_outlet["Points:1"].to_numpy()
vx_oldroyd = df_oldroyd_outlet["U:0"].to_numpy()

# Plot outlet velocities
fig2, ax2 = plt.subplots(1,1)
ax2.plot(vx_oldroyd[:int(np.floor(len(y_oldroyd)/2))], y_oldroyd[:int(np.floor(len(y_oldroyd)/2))], 'b', lw=2)
ax2.plot(vx_rude[:int(np.floor(len(y_rude)/2))], y_rude[:int(np.floor(len(y_rude)/2))], 'r', lw=2)
ax2.plot(vx_giesekus[int(np.floor(len(y_giesekus)/2)):], y_giesekus[int(np.floor(len(y_giesekus)/2)):], 'k', lw=2)
ax2.set_xlim([0,1.6])
ax2.set_ylim([-1,1])
ax2.tick_params(which="both", direction="in", top=True, right=True)

# Import data for the overshoot velocity
df_giesekus_overshoot = pd.read_csv("simdata/t10_U_overshoot_Giesekus.csv")
y_giesekus = df_giesekus_overshoot["Points:1"].to_numpy()
vx_giesekus = df_giesekus_overshoot["U:0"].to_numpy()

df_rude_overshoot = pd.read_csv("simdata/t10_U_overshoot_RUDE.csv")
y_rude = df_rude_overshoot["Points:1"].to_numpy()
vx_rude = df_rude_overshoot["U:0"].to_numpy()

df_oldroyd_overshoot = pd.read_csv("simdata/t10_U_overshoot_OldroydB.csv")
y_oldroyd = df_oldroyd_overshoot["Points:1"].to_numpy()
vx_oldroyd = df_oldroyd_overshoot["U:0"].to_numpy()

# Plot overshoot velocities
ax2.plot(vx_oldroyd[:int(np.floor(len(y_oldroyd)/2))], y_oldroyd[:int(np.floor(len(y_oldroyd)/2))], 'b--', lw=2)
ax2.plot(vx_rude[:int(np.floor(len(y_rude)/2))], y_rude[:int(np.floor(len(y_rude)/2))], 'r--', lw=2)
ax2.plot(vx_giesekus[int(np.floor(len(y_giesekus)/2)):], y_giesekus[int(np.floor(len(y_giesekus)/2)):], 'k--', lw=2)
ax2.set_xlim([0,1.6])
ax2.set_ylim([-1,1])
ax2.tick_params(which="both", direction="in", top=True, right=True)

# Import data for the velocity contours
df_giesekus = pd.read_csv("simdata/t10_U_Giesekus.csv")
x_giesekus = df_giesekus["Points:0"].to_numpy()[:25040]
y_giesekus = df_giesekus["Points:1"].to_numpy()[:25040]
vx_giesekus = df_giesekus["U:0"].to_numpy()[:25040]
vy_giesekus = df_giesekus["U:1"].to_numpy()[:25040]
v_giesekus = np.sqrt(vx_giesekus**2 + vy_giesekus**2)
Xg, Yg, Vg = make_mesh(x_giesekus, y_giesekus, v_giesekus, "top")

df_rude = pd.read_csv("simdata/t10_U_RUDE.csv")
x_rude = df_rude["Points:0"].to_numpy()[:25040]
y_rude = df_rude["Points:1"].to_numpy()[:25040]
vx_rude = df_rude["U:0"].to_numpy()[:25040]
vy_rude = df_rude["U:1"].to_numpy()[:25040]
v_rude = np.sqrt(vx_rude**2 + vy_rude**2)
Xr, Yr, Vr = make_mesh(x_rude, y_rude, v_rude, "bottom")

# Plot velocity contours
X = np.concatenate((Xr, Xg))
Y = np.concatenate((Yr, Yg))
V = np.concatenate((Vr, Vg))
fig3, ax3 = plt.subplots(1,1, figsize=(10,8))
CS = ax3.contourf(X, Y, V, 20, cmap="jet")
ax3.set_xlim([-7.5,5])
ax3.set_ylim([-5,5])
#fig3.colorbar(CS)
plt.axis("off")

# Get the streamlines
df_giesekus = pd.read_csv("simdata/t10_streamlines_Giesekus.csv")
x_giesekus = df_giesekus["Points:0"].to_numpy()
y_giesekus = -df_giesekus["Points:1"].to_numpy()
line_idx = df_giesekus.index[df_giesekus["IntegrationTime"] == 0].tolist()
sx_giesekus = []
sy_giesekus = []
for i in range(2,len(line_idx)):
    sx_giesekus += [x_giesekus[line_idx[i-1]:line_idx[i]]]
    sy_giesekus += [y_giesekus[line_idx[i-1]:line_idx[i]]]
cutoff = 55
sx_giesekus[0] = sx_giesekus[0][:cutoff]
sy_giesekus[0] = sy_giesekus[0][:cutoff]
sx_giesekus[0][-1] = sx_giesekus[0][0]
sy_giesekus[0][-1] = sy_giesekus[0][0]
ax3.plot(sx_giesekus[0],sy_giesekus[0],c="darkgray")
ax3.plot(sx_giesekus[4],sy_giesekus[4],c="darkgray")
for i in range(6,9):
    ax3.plot(sx_giesekus[i],sy_giesekus[i],c="darkgray")
ax3.plot(sx_giesekus[25],sy_giesekus[25],c="darkgray")
for i in range(27,30):
    ax3.plot(sx_giesekus[i],sy_giesekus[i],c="darkgray")

df_rude = pd.read_csv("simdata/t10_streamlines_RUDE.csv")
x_rude = df_rude["Points:0"].to_numpy()
y_rude = df_rude["Points:1"].to_numpy()
line_idx = df_rude.index[df_rude["IntegrationTime"] == 0].tolist()
sx_rude = []
sy_rude = []
for i in range(2,len(line_idx)):
    sx_rude += [x_rude[line_idx[i-1]:line_idx[i]]]
    sy_rude += [y_rude[line_idx[i-1]:line_idx[i]]]
cutoff = 50
sx_rude[0] = sx_rude[0][:cutoff]
sy_rude[0] = sy_rude[0][:cutoff]
sx_rude[0][-1] = sx_rude[0][0]
sy_rude[0][-1] = sy_rude[0][0]
ax3.plot(sx_rude[0],sy_rude[0],c="darkgray")
ax3.plot(sx_rude[4],sy_rude[4],c="darkgray")
for i in range(6,9):
    ax3.plot(sx_rude[i],sy_rude[i],c="darkgray")
ax3.plot(sx_rude[25],sy_rude[25],c="darkgray")
for i in range(27,30):
    ax3.plot(sx_rude[i],sy_rude[i],c="darkgray")

plt.show()

