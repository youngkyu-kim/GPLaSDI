#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt

nt = 201
tf = 2*np.pi
nx = 600

x = np.linspace(-3,3,nx)
t = np.linspace(0,tf,nt)

all_amp = [1.,1.2,1.4] # training parameters

param_test = np.asarray(all_amp)
n_test = len(param_test)

X_test = np.zeros([len(all_amp),nt, nx])

for ip in range(len(all_amp)):
    amp = all_amp[ip]
    for ix in range(0,nx):
        X_test[ip,:,ix] = amp*( np.sin(2*x[ix] - t) + 0.1*np.cos((40*x[ix] + 2*t)*np.sin(t)) )*np.exp(-x[ix]**2)


X_train = X_test[::2,:,:] + 0

param_train = param_test[::2]
n_train = len(param_train)


data_train = {'param_train' : param_train, 'X_train' : X_train, 
              'n_train' : n_train, 'x' : x, 't' : t}
os.makedirs(os.path.dirname("./data/data_train.npy"), exist_ok=True)
np.save('data/data_train.npy', data_train)

data_test = {'param_grid' : param_test, 'n_test' : n_test, 'X_test' : X_test,
             'x' : x, 't' : t}
os.makedirs(os.path.dirname("./data/data_test.npy"), exist_ok=True)
np.save('data/data_test.npy', data_test)


