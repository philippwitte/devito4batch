import sys, os
#sys.path.insert(0,'/home/pwitte/devito4batch/pysource/')
sys.path.insert(0,'/usr/local/devito4batch/pysource/')

import numpy as np
import matplotlib.pyplot as plt
from models import Model
from sources import RickerSource, TimeAxis, Receiver
from propagators import born, gradient, forward
from scipy import interpolate, ndimage
import matplotlib.pyplot as plt
import segyio
import mpi4py


#########################################################################################



def segy_write(data, sourceX, sourceZ, groupX, groupZ, dt, filename, sourceY=None, groupY=None, elevScalar=-1000, coordScalar=-1000):

    nt = data.shape[0]
    nxrec = len(groupX)
    if sourceY is None and groupY is None:
        sourceY = np.zeros(1, dtype='int')
        groupY = np.zeros(nxrec, dtype='int')

    # Create spec object
    spec = segyio.spec()
    spec.ilines = np.arange(nxrec)    # dummy trace count
    spec.xlines = np.zeros(1, dtype='int')  # assume coordinates are already vectorized for 3D
    spec.samples = range(nt)
    spec.format=1
    spec.sorting=1

    with segyio.create(filename, spec) as segyfile:
        for i in range(nxrec):
            segyfile.header[i] = {
                segyio.su.tracl : i+1,
                segyio.su.tracr : i+1,
                segyio.su.fldr : 1,
                segyio.su.tracf : i+1,
                segyio.su.sx : int(np.round(sourceX[0] * np.abs(coordScalar))),
                segyio.su.sy : int(np.round(sourceY[0] * np.abs(coordScalar))),
                segyio.su.selev: int(np.round(sourceZ[0] * np.abs(elevScalar))),
                segyio.su.gx : int(np.round(groupX[i] * np.abs(coordScalar))),
                segyio.su.gy : int(np.round(groupY[i] * np.abs(coordScalar))),
                segyio.su.gelev : int(np.round(groupZ[i] * np.abs(elevScalar))),
                segyio.su.dt : int(np.round(dt, decimals=6)*1e3),
                segyio.su.scalel : int(elevScalar),
                segyio.su.scalco : int(coordScalar)
            }
            segyfile.trace[i] = data[:, i]
        segyfile.dt=int(np.round(dt, decimals=6)*1e3)


def Ricker(f0, t):
    r = (np.pi * f0 * (t - 1./f0))
    w = (1-2.*r**2)*np.exp(-r**2)
    return w.reshape(len(w), 1)


def collect_shot(comm, rec):
    rank = comm.Get_rank()
    size = comm.size
    if (rank == 0):
        data = np.array(rec.data)
        coord = np.array(rec.coordinates_data)
        for i in range(1,size):
            datarcv = comm.recv(source=i, tag=0)
            coordrcv = comm.recv(source=i, tag=1)
            data = np.concatenate((data,datarcv),axis=1)
            coord = np.concatenate((coord,coordrcv),axis=0)
        tmp = np.concatenate((coord.transpose(),data))
        tmp = [tuple(tmp[:,tr]) for tr in range(tmp.shape[1])]
        tmp = sorted(tmp, key=lambda trace: trace[:3])
        tmp = np.array(tmp).transpose()
        data = tmp[3:,:]
        return data, coord                                                                                                                                        
    else:
        comm.send(np.array(rec.data) , dest=0, tag=0)
        comm.send(np.array(rec.coordinates_data) , dest=0, tag=1)
        return None, None
    

shape = (151, 151, 121)
rho = np.ones(shape, dtype='float32')
epsilon = 0.01*np.ones(shape, dtype='float32')
delta = 0.005*np.ones(shape, dtype='float32')
theta = 0.02*np.ones(shape, dtype='float32')
phi = 0.02*np.ones(shape, dtype='float32')

vp = 2*np.ones(shape, dtype='float32') 
vp[:, :, 60:] = 4.0
vp0 = 2*np.ones(shape, dtype='float32')

m = 1.0/vp**2
m0 = 1.0/vp0**2
dm = m - m0

origin = (0.0, 0.0, 0.0)
spacing = (12.5, 12.5, 12.5)
so = 12

# Source geometry
src_coords = np.empty((1, 3), dtype='float32')
src_coords[0, 0] = 937.0
src_coords[0, 1] = 937.0
src_coords[0, 2] = 22.5

# Receiver geometry
nxrec = 199
nyrec = 299

rec_coords_x = np.array(np.linspace(12.5, 1875.0, nxrec))
rec_coords_y = np.array(np.linspace(12.5, 1875.0, nyrec))
rec_x, rec_y = np.meshgrid(rec_coords_x, rec_coords_y)

rec_coords =  np.empty((nxrec*nyrec, 3), dtype='float32')
rec_coords[:, 0] = rec_x.reshape(-1)
rec_coords[:, 1] = rec_y.reshape(-1)
rec_coords[:, 2] = np.array(np.linspace(6., 6., len(rec_coords)))

# Model structure
model = Model(shape=shape, origin=origin, spacing=spacing, m=m0, space_order=so, nbpml=40,
    epsilon=epsilon, delta=delta, theta=theta, phi=phi, dm=dm)


comm = model.grid.distributor.comm
rank = comm.Get_rank()
size = comm.size

#########################################################################################

# Source wavelet
tn = 1000.
dt_shot = model.critical_dt
nt = int(tn/dt_shot)
time_s = np.linspace(0, tn, nt)
wavelet = Ricker(0.015, time_s)


#########################################################################################

# Devito operator
d_obs = born(model, src_coords, rec_coords, wavelet, save=False)[0]

u0 = forward(model, src_coords, rec_coords, wavelet, save=True, t_sub=8)[1]
grad_dist = gradient(model, d_obs, d_obs.coordinates, u0, isic=False)[0]


if rank > 0:
    # Send result to master
    comm.send(model.m.local_indices, dest=0, tag=10)
    comm.send(grad_dist.data, dest=0, tag=11)

else:   # Master
    # Initialize full array
    grad = np.empty(shape=model.m.shape_global, dtype='float32')
    grad[model.m.local_indices] = grad_dist.data

    # Collect gradients
    for j in range(1, size):
        local_indices = comm.recv(source=j, tag=10)
        glocal = comm.recv(source=j, tag=11)
        grad[local_indices] = glocal

    # Save file
    g.data.dump('gradient_mpi.dat')


#g.data.dump('grad')




#segy_write(d_obs.data, src_coords[:, 0], src_coords[:, 2], rec_coords[:, 0], rec_coords[:, 2], dt_shot, 'shot_3d.segy')

# d_gather, rec_gather = collect_shot(comm, d_obs)

# if rank == 0:
#     segy_write(d_gather, src_coords[:, 0], src_coords[:, 2], rec_coords[:, 0], rec_coords[:, 2], dt_shot, 'shot_n_2.segy',
#     sourceY=src_coords[:, 1], groupY=rec_coords[:, 1])




