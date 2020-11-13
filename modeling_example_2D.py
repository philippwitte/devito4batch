import sys, os
sys.path.insert(0,'/home/pwitte/devito4batch/pysource/')
#sys.path.insert(0, os.getcwd() + '/pysource')
import numpy as np
import matplotlib.pyplot as plt
from models import Model
from sources import RickerSource, TimeAxis, Receiver
from propagators import born, gradient, forward
from scipy import interpolate, ndimage
import matplotlib.pyplot as plt
import segyio
#import mpi4py


#########################################################################################

def Ricker(f0, t):
    r = (np.pi * f0 * (t - 1./f0))
    w = (1-2.*r**2)*np.exp(-r**2)
    return w.reshape(len(w), 1)
    
shape = (801, 267)
epsilon = 0.1*np.ones(shape, dtype='float32')
delta = 0.05*np.ones(shape, dtype='float32')
theta = np.pi/2*np.ones(shape, dtype='float32')

vp = 1.5*np.ones(shape, dtype='float32') 
vp[:, 100:] = 3.0
vp0 = 1.5*np.ones(shape, dtype='float32')

m = 1.0/vp**2
m0 = 1.0/vp0**2

origin = (0.0, 0.0)
spacing = (12.5, 12.5)
so = 12

# Source geometry
src_coords = np.empty((1, 2), dtype='float32')
src_coords[0, 0] = 4500.0
src_coords[0, 1] = 287.5

# Receiver geometry
nrec = 799
rec_coords =  np.empty((nrec, 2), dtype='float32')
rec_coords[:, 0] = np.array(np.linspace(12.5, 9987.5, nrec))
rec_coords[:, 1] = np.array(np.linspace(6., 6., nrec))

# Model structure
model = Model(shape=shape, origin=origin, spacing=spacing, m=m, space_order=so,
    epsilon=epsilon, delta=delta, theta=theta, nbpml=40)

model0 = Model(shape=shape, origin=origin, spacing=spacing, m=m0, space_order=so,
    epsilon=epsilon, delta=delta, theta=theta, nbpml=40)


# comm = model.grid.distributor.comm
# rank = comm.Get_rank()
# size = comm.size

#########################################################################################

# Source wavelet
tn = 1000.
dt_shot = model.critical_dt
nt = int(tn/dt_shot)
time_s = np.linspace(0, tn, nt)
wavelet = Ricker(0.015, time_s)


#########################################################################################

# Devito operator
d_obs = forward(model, src_coords, rec_coords, wavelet, save=False, t_sub=1)[0]

# Gradient
u0 = forward(model, src_coords, rec_coords, wavelet, save=True, t_sub=1)[1]
g = gradient(model, d_obs, d_obs.coordinates, u0)

