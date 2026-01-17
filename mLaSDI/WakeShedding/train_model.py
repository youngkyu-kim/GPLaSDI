#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import time
from lasdi.physics import Physics
from lasdi.latent_dynamics.sindy import SINDy
import os
import matplotlib.pyplot as plt
from models import Autoencoder, CorrectingDecoder

#make plots pretty
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fontsize'] = 20
# plt.rcParams['suptitle.fontsize'] = 20
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['figure.dpi'] = 150

### Main


date = time.localtime()
date_str = "{month:02d}_{day:02d}_{year:04d}_{hour:02d}_{minute:02d}"
date_str = date_str.format(month = date.tm_mon, day = date.tm_mday, year = date.tm_year, hour = date.tm_hour + 3, minute = date.tm_min)

name = date_str

#Hyperparameters 
# dt = 0.001 #CHANGE THIS IF WE CHANGE TIMESTEP OF DATA
dt = 0.01
ae_weight = 1e0
sindy_weight = 1.e-1
coef_weight= 1.e-3
lr = 1e-3


path_data = 'data/'
data_train = np.load(path_data + 'data_train.npy', allow_pickle = True).item()
data_test = np.load(path_data + 'data_test.npy', allow_pickle = True).item()
X_test = data_test['X_test']

X_train = torch.Tensor(data_train['X_train'])

# we subsample time just to make training a bit faster
X_train = X_train[:,::10,:]
X_test = X_test[:,::10,:]

param_grid = data_test['param_grid']


sol_dim = 1
time_dim = X_train.shape[1]
nx = X_train.shape[-1]

t_grid = np.linspace(0, (time_dim - 1)*dt, time_dim)

class CustomPhysicsModel(Physics):
    def __init__(self):
        self.dim = 1
        self.nt = time_dim
        self.dt = dt
        self.qdim = sol_dim
        self.grid_size = nx
        self.qgrid_size = [nx]
        self.t_grid = t_grid

        return
    
physics = CustomPhysicsModel()


#Architecture
hidden_units = [50]
n_z = 5

autoencoder = Autoencoder(nx, hidden_units, n_z)

sindy_options = {'sindy': {'fd_type': 'sbp12', 'coef_norm_order': 2} } # finite-difference operator for computing time derivative of latent trajectory.
# sindy_options = {'sindy': {'fd_type': 'sbp12', 'coef_norm_order': torch.inf} } # finite-difference operator for computing time derivative of latent trajectory.
ld = SINDy(autoencoder.n_z, physics.nt, sindy_options)


#Setup training
best_loss = np.inf

optimizer = torch.optim.Adam(autoencoder.parameters(), lr = lr)
MSE = torch.nn.MSELoss()

cuda = torch.cuda.is_available()
if cuda:
    device = 'cuda'
else:
    device = 'cpu'
    
    
print(device)
        
d_ae = autoencoder.to(device)
d_Xtrain = X_train.to(device)

n_iter = 2500
n_train = X_train.shape[0]
save_interval = 1
hist_file = '%s.loss_history.txt' % name

loss_hist = np.zeros([n_iter, 4])
grad_hist = np.zeros([n_iter, 4])
 
tic_start = time.time()

for iter in range(n_iter):
    optimizer.zero_grad()
    # d_ae = ae.to(device)
    d_Z = d_ae.encoder(d_Xtrain)
    d_Xpred = d_ae.decoder(d_Z)
    Z = d_Z.cpu()

    loss_ae = MSE(d_Xtrain, d_Xpred)
    coefs, loss_sindy, loss_coef = ld.calibrate(Z, physics.dt, compute_loss=True, numpy=False)
 
    max_coef = np.abs(np.array(coefs)).max()
    loss = ae_weight * loss_ae + sindy_weight * loss_sindy / n_train + coef_weight * loss_coef / n_train

    loss_hist[iter] = [loss.item(), loss_ae.item(), loss_sindy.item(), loss_coef.item()]

    loss.backward()

    optimizer.step()

    if ((loss.item() < best_loss) and (iter % save_interval == 0)):
        os.makedirs(os.path.dirname('checkpoint/' + './%s_checkpoint.pt' % name), exist_ok=True)
        torch.save(d_ae.cpu().state_dict(), 'checkpoint/%s_checkpoint.pt' % name)
        d_ae = autoencoder.to(device)
        best_loss = loss.item()
        best_coefs = coefs

    # print("Iter: %05d/%d, Loss: %.5e" % (iter + 1, n_iter, loss.item()))
    print("Iter: %05d/%d, Loss: %.5e, Loss AE: %.5e, Loss SI: %.5e, Loss COEF: %.5e, max|c|: %.5e"
            % (iter + 1, n_iter, loss.item(), loss_ae.item(), loss_sindy.item(), loss_coef.item(), max_coef))

tic_end = time.time()
total_time = tic_end - tic_start



os.makedirs(os.path.dirname('losses/' + './%s_checkpoint.pt' % name), exist_ok=True)
np.savetxt('losses/%s.loss_history.txt' % name, loss_hist)

if (loss.item() < best_loss):
    torch.save(d_ae.cpu().state_dict(), 'checkpoint/%s_checkpoint.pt' % name)
    best_loss2 = loss.item()
else:
    d_ae.cpu().load_state_dict(torch.load('checkpoint/%s_checkpoint.pt' % name))
    
param_train = data_train['param_train']
param_test = data_test['param_grid']

bglasdi_results = {'autoencoder_param': d_ae.cpu().state_dict(), 'final_param_train': param_train,
                            'final_X_train': X_train, 'param_test': param_test,
                            'sindy_coef': best_coefs, 'gp_dictionnary': None, 'lr': lr, 'n_iter': n_iter,
                            'sindy_weight': sindy_weight, 'coef_weight': coef_weight,
                            't_grid' : t_grid, 'dt' : dt,'total_time' : total_time}
    
os.makedirs(os.path.dirname('results/'), exist_ok=True)
np.save('results/bglasdi_' + date_str + '_lr' + str(lr) + '_sw' + str(sindy_weight) + '_cw'+str(coef_weight)+'_nt' + str(time_dim) + '_niter' + str(n_iter) + '_nz' + str(n_z) +'.npy', bglasdi_results)

fig = plt.figure()
ax = plt.axes()
ax.set_title('First Stage Losses')
ax.plot(loss_hist[:,1],label = 'AE')
ax.plot(loss_hist[:,2],label = 'LD')
ax.plot(loss_hist[:,3],label = 'Coeff')
ax.legend()
ax.set_yscale('log')
ax.set_xlabel('Iterations')

fig = plt.figure()
ax = plt.axes()
ax.set_title('First Stage Loss')
ax.plot(loss_hist[:,0],label = 'Total Loss')
#ax.legend()
ax.set_yscale('log')
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')




# %% plotting

autoencoder_param = autoencoder.cpu().state_dict()
AEparams = [value for value in autoencoder_param.items()]
AEparamnum = 0
for i in range(len(AEparams)):
    AEparamnum = AEparamnum + (AEparams[i][1].detach().numpy()).size

from lasdi.gp import fit_gps
from lasdi.gplasdi import sample_roms, average_rom
from lasdi.postprocess import compute_errors
from lasdi.gp import eval_gp
import numpy.linalg as LA

# all_kinvis = [0.00064, 0.00043, 0.00032]
all_kinvis = [0.00054, 0.00043, 0.00032]


def IC(kinvis):
    # print(kinvis)
    for i in range(X_test.shape[0]):
        if np.abs(kinvis[0] - all_kinvis[i]) <1E-8:
            initcond = X_test[i,0,:]
            break
    return initcond

# def IC(kinvis):
#     # print(kinvis)
#     for i in range(X_test.shape[0]):
#         if np.abs(kinvis[0] - (i+1)) <1E-8:
#             initcond = X_test[i,0,:]
#             break
#     return initcond

physics.initial_condition = IC

meanparam = param_train.mean(0)
stdparam = param_train.std(0)

# meanparam = param_train.max(0)
# stdparam = param_train.max(0) - param_train.min(0)

# param_train_normal = ( param_train - meanparam ) / stdparam
# param_grid_normal = ( param_grid - meanparam ) / stdparam

param_train_normal = param_train
param_grid_normal = param_grid

# param_train_normal = np.asarray([1,3])
# param_grid_normal = np.asarray([1,2,3])

param_train_normal = np.vstack(( param_train_normal, np.ones(len(param_train_normal)) )).T
param_grid_normal = np.vstack((param_grid_normal, np.ones(len(param_grid_normal)) )).T


# n_samples = 20
autoencoder = d_ae.cpu()

Z = autoencoder.encoder(X_train)

coefs = ld.calibrate(Z, physics.dt, compute_loss=False, numpy=True)

gp_dictionnary = fit_gps(param_train_normal, coefs)
# gp_dictionnary = fit_gps(param_train, coefs,normalize = True)

# Zis_samples = sample_roms(autoencoder, physics, ld, gp_dictionnary, param_grid_normal, n_samples)
Zis_mean = average_rom(autoencoder, physics, ld, gp_dictionnary, param_grid_normal)

# Zis_samples = sample_roms(autoencoder, physics, ld, gp_dictionnary, param_grid, n_samples)
# Zis_mean = average_rom(autoencoder, physics, ld, gp_dictionnary, param_grid)

X_pred_mean = autoencoder.decoder(torch.Tensor(Zis_mean)).detach().numpy()
# X_pred_samples = autoencoder.decoder(torch.Tensor(Zis_samples)).detach().numpy()

X_AE_recon = autoencoder(X_train).detach().numpy()

Zis_mean_train = torch.Tensor(average_rom(autoencoder, physics, ld, gp_dictionnary, param_train_normal)).detach()
X_pred_mean_train = autoencoder.decoder(torch.Tensor(Zis_mean_train)).detach()

# %% 

param_ind = 2
amp = param_grid[param_ind]
param = np.array([[amp]])

# X_test = data_test['X_test']

true = X_test[param_ind,:,:]

Z_mean = Zis_mean[param_ind,:,:]

true_traj = autoencoder.encoder(torch.Tensor(true))
true_traj = true_traj.detach().numpy()
fig = plt.figure()
ax = plt.axes()
ax.set_title('Latent Space Trajectories')
#fig.suptitle(f'Prediction, Standoff = {stand:.2f} cm, Charge radius = {charge:.2f} cm', y = 1.05)
ax.plot(t_grid,true_traj)
ax.plot(t_grid,Z_mean,'--')
ax.set_xlabel('Time')


AE_recon = autoencoder.decoder(autoencoder.encoder(torch.Tensor(true)))
AE_recon = AE_recon.detach().numpy()

pred_mean = autoencoder.decoder(torch.Tensor(Z_mean)).detach().numpy()

# maxnorm = np.max(LA.norm(true,axis=1));    
maxnorm = LA.norm(true,axis=1)
recon_error_AE = LA.norm(AE_recon-true,axis=1)/maxnorm*100
recon_error_lasdi = LA.norm(pred_mean-true,axis=1)/maxnorm*100
 
l = [hidden_units,[n_z]]
flat_list = [item for sublist in l for item in sublist]

fig = plt.figure()
#fig.suptitle(f'Prediction, Standoff = {stand:.2f} cm, Charge radius = {charge:.2f} cm', y = 1.05)
ax = plt.axes()
#ax.set_title('Max Relative Error: {:.3}\%'.format(np.amax(recon_error_lasdi)))
ax.plot(t_grid,recon_error_AE, label = 'Autoencoder')
ax.plot(t_grid,recon_error_lasdi,'--', label = 'GPLaSDI')
ax.set_xlabel('Time')
ax.set_ylabel('Relative Error (\%)')
ax.legend()
# ax.set_ylim(0,100)
# ax.set_ylim(1E-2,1500)
ax.set_yscale('log')




# %% second setup


err = (d_Xtrain - torch.Tensor(X_pred_mean[::2,:,:]).to(device) ).detach()
d_Xpred2 = err/(torch.std(err))
model = CorrectingDecoder(n_z, hidden_units[::-1], nx,kappa=1)

best_loss2 = np.inf

optimizer2 = torch.optim.Adam(model.parameters(), lr = lr)
MSE2 = torch.nn.MSELoss()
   

d_model = model.to(device)




n_iter = 2500
n_train = X_train.shape[0]
save_interval = 1
hist_file = '%s.loss_history.txt' % name

loss_hist2 = np.zeros([n_iter, 1])
 
tic_start = time.time()



for iter in range(n_iter):
    optimizer2.zero_grad()
    d_Xpred = d_model(Zis_mean_train)

    loss = MSE2(d_Xpred,d_Xpred2)

    loss_hist2[iter] = [loss.item()]

    loss.backward()

    optimizer2.step()

    if ((loss.item() < best_loss2) and (iter % save_interval == 0)):
        os.makedirs(os.path.dirname('checkpoint/' + './%s_checkpoint.pt' % name), exist_ok=True)
        torch.save(d_model.cpu().state_dict(), 'checkpoint/%s_checkpoint.pt' % name)
        d_model = model.to(device)
        best_loss2 = loss.item()

    print("Iter: %05d/%d, Loss: %.5e"
            % (iter + 1, n_iter, loss.item()))

tic_end = time.time()
total_time2 = tic_end - tic_start

if (loss.item() < best_loss2):
    torch.save(d_model.cpu().state_dict(), 'checkpoint/%s_checkpoint.pt' % name)
    best_loss2 = loss.item()
else:
    d_model.cpu().load_state_dict(torch.load('checkpoint/%s_checkpoint.pt' % name))

bglasdi_results2 = {'model_param': model.cpu().state_dict(), 'final_param_train': param_train,
                            'final_X_train': X_train, 'param_test': param_test,
                            'sindy_coef': best_coefs, 'gp_dictionnary': None, 'lr': lr, 'n_iter': n_iter,
                            'sindy_weight': sindy_weight, 'coef_weight': coef_weight,
                            't_grid' : t_grid, 'dt' : dt,'total_time' : total_time}

os.makedirs(os.path.dirname('results/'), exist_ok=True)
np.save('results/bglasdicorr_' + date_str + '_lr' + str(lr) + '_sw' + str(sindy_weight) + '_cw'+str(coef_weight)+'_nt' + str(time_dim) + '_niter' + str(n_iter) + '_nz' + str(n_z) +'.npy', bglasdi_results)


fig = plt.figure()
ax = plt.axes()
ax.set_title('Second Stage Total Loss')
ax.plot(loss_hist2[:,0],label = 'Total Loss')
ax.legend()
ax.set_yscale('log')
ax.set_xlabel('Iterations')

# %% Second stage results

model = d_model.cpu()

X_pred_mean2 = model(torch.Tensor(Zis_mean)).detach().numpy()

model_param = model.cpu().state_dict()
modelparams = [value for value in model_param.items()]
modelparamnum = 0
for i in range(len(modelparams)):
    modelparamnum = modelparamnum + (modelparams[i][1].detach().numpy()).size


# %% Plot second stage

param_ind = 2
amp = param_grid[param_ind]
param = np.array([[amp]])

true = X_test[param_ind,:,:]

AE_recon = autoencoder.decoder(autoencoder.encoder(torch.Tensor(true)))
AE_recon = AE_recon.detach().numpy()

Z_mean = Zis_mean[param_ind,:,:]

pred_mean = autoencoder.decoder(torch.Tensor(Z_mean)).detach().numpy()

denom = LA.norm(true,axis=1)
recon_error_AE = LA.norm(AE_recon-true,axis=1)/denom*100
recon_error_lasdi = LA.norm(pred_mean-true,axis=1)/denom*100
 
fig = plt.figure()
ax = plt.axes()
ax.plot(t_grid,recon_error_AE, label = 'Autoencoder')
ax.plot(t_grid,recon_error_lasdi,'--', label = 'GPLaSDI')
ax.set_xlabel('Time')
ax.set_ylabel('Relative Error (\%)')
ax.legend()
ax.set_yscale('log')

stdval2 = (torch.std(err.cpu())).detach().numpy()

pred_mean2 = model(torch.Tensor(Z_mean)).detach().numpy()*stdval2

recon_error_lasdi2 = LA.norm(pred_mean + pred_mean2 - true,axis=1)/denom*100
 

fig = plt.figure()
ax = plt.axes()
ax.set_title(f'$A =  ${param_test[param_ind]:.1f}')
ax.plot(t_grid,recon_error_lasdi, label = 'First Stage')
ax.plot(t_grid,recon_error_lasdi2,'--', label = 'Second Stage')
ax.set_xlabel('Time')
ax.set_ylabel('Relative Error (\%)')
ax.legend()
ax.set_yscale('log')


# %% all errs
fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)

fig.set_size_inches(8, 3)

for i in range(3):
    param_ind = i
    amp = param_grid[param_ind]
    param = np.array([[amp]])

    true = X_test[param_ind,:,:]

    Z_mean = Zis_mean[param_ind,:,:]



    AE_recon = autoencoder.decoder(autoencoder.encoder(torch.Tensor(true)))
    AE_recon = AE_recon.detach().numpy()

    pred_mean = autoencoder.decoder(torch.Tensor(Z_mean)).detach().numpy()

    # maxnorm = np.max(LA.norm(true,axis=1));    
    maxnorm = LA.norm(true,axis=1)
    recon_error_AE = LA.norm(AE_recon-true,axis=1)/maxnorm*100
    recon_error_lasdi = LA.norm(pred_mean-true,axis=1)/maxnorm*100

    ###
    stdval2 = (torch.std(err.cpu())).detach().numpy()

    pred_mean2 = model(torch.Tensor(Z_mean)).detach().numpy()*stdval2

    recon_error_lasdi2 = LA.norm(pred_mean + pred_mean2 - true,axis=1)/maxnorm*100


    if i == 1:
        ax[i].set_title(f'$\\nu =  ${param_test[param_ind]:.5f}',color='r')
    else:
        ax[i].set_title(f'$\\nu =  ${param_test[param_ind]:.5f}')
    ax[i].plot(t_grid,recon_error_lasdi,linewidth=1.5, label = 'First Stage')
    ax[i].plot(t_grid,recon_error_lasdi2,'--', linewidth=1.5, label = 'Second Stage')
    ax[i].set_xlabel('Time')
    if i == 0:
        ax[i].set_ylabel('Relative Error (\%)')
        lines_labels = ax[i].get_legend_handles_labels()
        lines, labels = lines_labels
    ax[i].set_yscale('log')
fig.legend(lines, labels,loc='upper center',ncol=2,bbox_to_anchor=(0.525, -0.125))  



