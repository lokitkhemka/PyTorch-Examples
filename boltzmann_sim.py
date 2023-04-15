import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from matplotlib.animation import PillowWriter



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#DEFINING THE NUMBER OF PARTICLES AND GENERATE THEIR POSITIONS AT RANDOM (BETWEEN 0 AND 1)
n_particles = 10000
r = torch.rand((2,n_particles)).to(device)

ixr = r[0]>0.5 #Right
ixl = r[0]<0.5 #Left

ids = torch.arange(n_particles)


#ASSIGNING THE VELOCITIES TO EACH PARTICLES
v = torch.zeros((2,n_particles)).to(device)
v[0][ixr] = -500 #Assigning a velocity of -500 to particles on the right side
v[0][ixl] = 500

#DISTANCE BETWEEN EACH PARTICLE PAIRS
ids_pairs = torch.combinations(ids,2).to(device)
x_pairs = torch.combinations(r[0],2).to(device)
y_pairs = torch.combinations(r[1],2).to(device)
dx_pairs = torch.diff(x_pairs, axis = 1).ravel()
dy_pairs = torch.diff(y_pairs, axis = 1).ravel()

d_pairs = torch.sqrt(dx_pairs**2 + dy_pairs**2)

#Check for collisions
radius = 0.003
ids_pairs_collide = ids_pairs[d_pairs < 2* radius]

v1 = v[:,ids_pairs_collide[:,0]]
v2 = v[:,ids_pairs_collide[:,1]]
r1 = r[:,ids_pairs_collide[:,0]]
r2 = r[:,ids_pairs_collide[:,1]]


def get_deltad_pairs(r):
   dx = torch.diff(torch.combinations(r[0],2).to(device)).squeeze()
   dy = torch.diff(torch.combinations(r[1],2).to(device)).squeeze()
   return torch.sqrt(dx**2 + dy**2)

def compute_new_v(v1,v2,r1,r2):
    v1new = v1 - torch.sum((v1-v2)*(r1-r2), axis=0)/torch.sum((r1-r2)**2, axis = 0) * (r1-r2)
    v2new = v2 - torch.sum((v1-v2)*(r1-r2), axis=0)/torch.sum((r2-r1)**2, axis = 0) * (r2-r1)
    return v1new, v2new

def motion(r,v, id_pairs, ts, dt, d_cutoff):
    rs = torch.zeros((ts, r.shape[0], r.shape[1])).to(device)
    vs = torch.zeros((ts, v.shape[0], v.shape[1])).to(device)

    #Intial State
    rs[0] = r
    vs[0] = v
    for i in range(1,ts):
        ic = id_pairs[get_deltad_pairs(r)<d_cutoff]
        v[:,ic[:,0]], v[:,ic[:,1]] = compute_new_v(v[:,ic[:,0]],v[:,ic[:,1]],r[:,ic[:,0]],r[:,ic[:,1]])
        #Box Collisions
        v[0,r[0]>1] = -torch.abs(v[0,r[0]>1])
        v[0,r[0]<0] = torch.abs(v[0,r[0]<0])
        v[1,r[1]>1] = -torch.abs(v[1,r[1]>1])
        v[1,r[1]<0] = torch.abs(v[1,r[1]<0])

        r = r + v*dt
        rs[i] = r
        vs[i] = v
    return rs, vs

rs,vs = motion(r,v,ids_pairs, ts=1000, dt = 0.000008, d_cutoff = 2*radius)

#Plotting and Animation
v = np.linspace(0, 2000, 1000)
a = 2/500**2
fv = a*v*np.exp(-a*v**2 / 2)


bins = np.linspace(0,1500,50)
plt.figure()
plt.hist(torch.sqrt(torch.sum(vs[400]**2, axis=0)).cpu(), bins=bins, density=True)
plt.plot(v,fv)
plt.xlabel('Velocity [m/s]')
plt.ylabel('# Particles')



fig, axes = plt.subplots(1, 2, figsize=(20,10))
axes[0].clear()
vmin = 0
vmax = 1
axes[0].set_xlim(0,1)
axes[0].set_ylim(0,1)
markersize = 2 * radius * axes[0].get_window_extent().width  / (vmax-vmin) * 72./fig.dpi
red, = axes[0].plot([], [], 'o', color='red', markersize=markersize)
blue, = axes[0].plot([], [], 'o', color='blue', markersize=markersize)
n, bins, patches = axes[1].hist(torch.sqrt(torch.sum(vs[0]**2, axis=0)).cpu(), bins=bins, density=True)
axes[1].plot(v,fv)
axes[1].set_ylim(top=0.003)

def animate(i):
    xred, yred = rs[i][0][ixr].cpu(), rs[i][1][ixr].cpu()
    xblue, yblue = rs[i][0][ixl].cpu(),rs[i][1][ixl].cpu()
    red.set_data(xred, yred)
    blue.set_data(xblue, yblue)
    hist, _ = np.histogram(torch.sqrt(torch.sum(vs[i]**2, axis=0)).cpu(), bins=bins, density=True)
    for i, patch in enumerate(patches):
        patch.set_height(hist[i])
    return red, blue

writer = animation.FFMpegWriter(fps=30)
ani = animation.FuncAnimation(fig, animate, frames=500, interval=50, blit=True)
ani.save('./boltzmann_sim.mp4',writer=writer,dpi=100)

