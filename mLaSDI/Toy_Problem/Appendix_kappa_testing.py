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

torch.manual_seed(0)
np.random.rand(0)

    
    
date = time.localtime()
date_str = "{month:02d}_{day:02d}_{year:04d}_{hour:02d}_{minute:02d}"
date_str = date_str.format(month = date.tm_mon, day = date.tm_mday, year = date.tm_year, hour = date.tm_hour + 3, minute = date.tm_min)

name = date_str


ae_weight = 1e0
sindy_weight = 1.e-1
coef_weight= 1.e-3
lr = 1e-3


path_data = 'data/'
data_train = np.load(path_data + 'data_train.npy', allow_pickle = True).item()
data_test = np.load(path_data + 'data_test.npy', allow_pickle = True).item()

X_train = torch.Tensor(data_train['X_train'])

param_grid = data_test['param_grid']

sol_dim = 1
grid_dim = [data_test['param_grid'].shape[0]]
time_dim = X_train.shape[1]
nx = X_train.shape[-1]



x = np.linspace(-3,3,nx)
dt = 2*np.pi/X_train.shape[1]
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
    
    ''' See lasdi.physics.Physics class for necessary subroutines '''
    
physics = CustomPhysicsModel()

# Parameter Space Definition


hidden_units = [200,20]
n_z = 10

autoencoder = Autoencoder(nx, hidden_units, n_z)

# %%
best_loss = np.inf

sindy_options = {'sindy': {'fd_type': 'sbp12', 'coef_norm_order': 2} } # finite-difference operator for computing time derivative of latent trajectory.
ld = SINDy(autoencoder.n_z, physics.nt, sindy_options)

# optimizer = torch.optim.Adam(autoencoder.parameters(), lr = .1*lr)
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

n_iter = 10000
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
    best_loss = loss.item()
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

def IC(amp):
    return amp[0]*( np.sin(2*x) + 0.1 )*np.exp(-x**2)

physics.initial_condition = IC

meanparam = param_train.mean(0)
stdparam = param_train.std(0)

param_train_normal = param_train
param_grid_normal = param_grid

# need to make parameters at least 2D.This is not always true
param_train_normal = np.vstack(( param_train_normal, np.ones(len(param_train_normal)) )).T
param_grid_normal = np.vstack((param_grid_normal, np.ones(len(param_grid_normal)) )).T


# n_samples = 20
autoencoder = d_ae.cpu()

Z = autoencoder.encoder(X_train)

coefs = ld.calibrate(Z, physics.dt, compute_loss=False, numpy=True)

gp_dictionnary = fit_gps(param_train_normal, coefs)

Zis_mean = average_rom(autoencoder, physics, ld, gp_dictionnary, param_grid_normal)

X_pred_mean = autoencoder.decoder(torch.Tensor(Zis_mean)).detach().numpy()

Zis_mean_train = torch.Tensor(average_rom(autoencoder, physics, ld, gp_dictionnary, param_train_normal)).detach()
X_pred_mean_train = autoencoder.decoder(Zis_mean_train).detach()


# %% second training, learn new decoder only - KAPPA SWEEP

kappa_values = [0.01,0.025,0.05,0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10,25, 50,75]
models_dict = {}  # Store trained models for each kappa
loss_histories = {}  # Store loss histories

err = (d_Xtrain - torch.Tensor(X_pred_mean[::2,:,:]).to(device)).detach()
d_Xpred2 = err / torch.std(err)
errcpu = err.cpu().numpy()
stdval2 = (torch.std(err.cpu())).detach().numpy()

n_iter = 10000
n_train = X_train.shape[0]
save_interval = 1

for kappa in kappa_values:
    print(f"\n{'='*50}")
    print(f"Training with kappa = {kappa}")
    print(f"{'='*50}\n")
    
    model = CorrectingDecoder(n_z, hidden_units[::-1], nx, kappa=kappa)
    best_loss2 = np.inf
    optimizer2 = torch.optim.Adam(model.parameters(), lr=lr)
    MSE2 = torch.nn.MSELoss()
    d_model = model.to(device)
    
    loss_hist2 = np.zeros([n_iter, 1])
    tic_start = time.time()
    
    for iter in range(n_iter):
        optimizer2.zero_grad()
        d_Xpred = d_model(Zis_mean_train)
        loss = MSE2(d_Xpred, d_Xpred2)
        loss_hist2[iter] = [loss.item()]
        loss.backward()
        optimizer2.step()
        
        if (loss.item() < best_loss2) and (iter % save_interval == 0):
            os.makedirs(os.path.dirname(f'checkpoint/{name}_kappa{kappa}_checkpoint.pt'), exist_ok=True)
            torch.save(d_model.cpu().state_dict(), f'checkpoint/{name}_kappa{kappa}_checkpoint.pt')
            d_model = model.to(device)
            best_loss2 = loss.item()
        
        if (iter + 1) % 1000 == 0:
            print(f"Iter: {iter+1:05d}/{n_iter}, Loss: {loss.item():.5e}")
    
    tic_end = time.time()
    print(f"Training time for kappa={kappa}: {tic_end - tic_start:.2f}s")
    
    # Load best model
    if loss.item() < best_loss2:
        torch.save(d_model.cpu().state_dict(), f'checkpoint/{name}_kappa{kappa}_checkpoint.pt')
    else:
        d_model.cpu().load_state_dict(torch.load(f'checkpoint/{name}_kappa{kappa}_checkpoint.pt'))
    
    models_dict[kappa] = d_model.cpu()
    loss_histories[kappa] = loss_hist2

# %% Plotting relative errors for all kappa values

fig, axes = plt.subplots(nrows=len(kappa_values), ncols=3, sharex=True,sharey=True, figsize=(12, 3*len(kappa_values)))

colors = plt.cm.viridis(np.linspace(0, 0.9, len(kappa_values)))

for k_idx, kappa in enumerate(kappa_values):
    model = models_dict[kappa]
    
    for i in range(3):
        param_ind = i
        amp = param_grid[param_ind]
        X_test = data_test['X_test']
        true = X_test[param_ind, :, :]
        Z_mean = Zis_mean[param_ind, :, :]
        
        pred_mean = autoencoder.decoder(torch.Tensor(Z_mean)).detach().numpy()
        maxnorm = LA.norm(true, axis=1)
        recon_error_lasdi = LA.norm(pred_mean - true, axis=1) / maxnorm * 100
        
        pred_mean2 = model(torch.Tensor(Z_mean)).detach().numpy() * stdval2
        recon_error_lasdi2 = LA.norm(pred_mean + pred_mean2 - true, axis=1) / maxnorm * 100
        
        ax = axes[k_idx, i]
        ax.plot(t_grid, recon_error_lasdi, '-', linewidth=1.5, label='First Stage')
        ax.plot(t_grid, recon_error_lasdi2, '--', linewidth=1.5, label='Second Stage')
        ax.set_yscale('log')
        
        if k_idx == 0:
            if i == 1:
                ax.set_title(f'$A = {param_test[param_ind]:.1f}$', color='r')
            else:
                ax.set_title(f'$A = {param_test[param_ind]:.1f}$')
        
        if i == 0:
            ax.set_ylabel(f'$\\kappa = {kappa}$\nRel. Error (\\%)')
        
        if k_idx == len(kappa_values) - 1:
            ax.set_xlabel('Time')
        
        if k_idx == 0 and i == 2:
            ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(f'results/kappa_sweep_comparison_{date_str}.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %% Alternative: Overlay plot comparing all kappas for each parameter

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))

cmap = plt.cm.plasma
colors = [cmap(i / (len(kappa_values) - 1)) for i in range(len(kappa_values))]

for i in range(3):
    param_ind = i
    X_test = data_test['X_test']
    true = X_test[param_ind, :, :]
    Z_mean = Zis_mean[param_ind, :, :]
    
    pred_mean = autoencoder.decoder(torch.Tensor(Z_mean)).detach().numpy()
    maxnorm = LA.norm(true, axis=1)
    recon_error_lasdi = LA.norm(pred_mean - true, axis=1) / maxnorm * 100
    
    # Plot first stage (same for all kappas)
    axes[i].plot(t_grid, recon_error_lasdi, 'k-', linewidth=2, label='First Stage', alpha=0.7)
    
    # Plot second stage for each kappa
    for k_idx, kappa in enumerate(kappa_values):
        model = models_dict[kappa]
        pred_mean2 = model(torch.Tensor(Z_mean)).detach().numpy() * stdval2
        recon_error_lasdi2 = LA.norm(pred_mean + pred_mean2 - true, axis=1) / maxnorm * 100
        axes[i].plot(t_grid, recon_error_lasdi2, '--', color=colors[k_idx], 
                     linewidth=1.5, label=f'$\\kappa = {kappa}$')
    
    axes[i].set_yscale('log')
    axes[i].set_xlabel('Time')
    if i == 1:
        axes[i].set_title(f'$A = {param_test[param_ind]:.1f}$', color='r', fontsize=14)
    else:
        axes[i].set_title(f'$A = {param_test[param_ind]:.1f}$', fontsize=14)
    
    if i == 0:
        axes[i].set_ylabel('Relative Error (\\%)')

# Single legend for all subplots
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=10)

plt.tight_layout()
plt.savefig(f'results/kappa_overlay_comparison_{date_str}.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %% Loss history comparison

fig, ax = plt.subplots(figsize=(8, 5))

for k_idx, kappa in enumerate(kappa_values):
    ax.plot(loss_histories[kappa][:, 0], label=f'$\\kappa = {kappa}$', 
            color=colors[k_idx], linewidth=1.5)

ax.set_yscale('log')
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.set_title('Second Stage Training Loss for Different $\\kappa$ Values')
ax.legend()
plt.tight_layout()
plt.savefig(f'results/kappa_loss_comparison_{date_str}.pdf', bbox_inches='tight', dpi=300)
plt.show()


# %% Compute min/max errors for each kappa

train_indices = [0, 2]
test_indices = [1]

# Storage for errors
# First stage (same for all kappa)
first_stage_train_min = []
first_stage_train_max = []
first_stage_test_min = []
first_stage_test_max = []

first_stage_train_mean = []
first_stage_test_mean = []

# Second stage (varies with kappa)
second_stage_train_min = {kappa: [] for kappa in kappa_values}
second_stage_train_max = {kappa: [] for kappa in kappa_values}
second_stage_test_min = {kappa: [] for kappa in kappa_values}
second_stage_test_max = {kappa: [] for kappa in kappa_values}

second_stage_train_mean = {kappa: [] for kappa in kappa_values}
second_stage_test_mean = {kappa: [] for kappa in kappa_values}

X_test = data_test['X_test']

# Compute first stage errors (independent of kappa)
for idx in train_indices + test_indices:
    true = X_test[idx, :, :]
    Z_mean = Zis_mean[idx, :, :]
    pred_mean = autoencoder.decoder(torch.Tensor(Z_mean)).detach().numpy()
    maxnorm = LA.norm(true, axis=1)
    recon_error = LA.norm(pred_mean - true, axis=1) / maxnorm * 100
    
    if idx in train_indices:
        first_stage_train_min.append(np.min(recon_error))
        first_stage_train_max.append(np.max(recon_error))
        first_stage_train_mean.append(np.mean(recon_error))
    else:
        first_stage_test_min.append(np.min(recon_error))
        first_stage_test_max.append(np.max(recon_error))
        first_stage_test_mean.append(np.mean(recon_error))

# Aggregate first stage across train/test sets
fs_train_min = np.min(first_stage_train_min)
fs_train_max = np.max(first_stage_train_max)
fs_test_min = np.min(first_stage_test_min)
fs_test_max = np.max(first_stage_test_max)

fs_train_mean = np.mean(first_stage_train_mean)
fs_test_mean = np.mean(first_stage_test_mean)

# Compute second stage errors for each kappa
results = {'kappa': [], 'train_min': [], 'train_max': [], 'train_mean': [], 'test_min': [], 'test_max': [], 'test_mean': []}

for kappa in kappa_values:
    model = models_dict[kappa]
    train_errors_all = []
    test_errors_all = []
    
    for idx in train_indices + test_indices:
        true = X_test[idx, :, :]
        Z_mean = Zis_mean[idx, :, :]
        pred_mean = autoencoder.decoder(torch.Tensor(Z_mean)).detach().numpy()
        pred_mean2 = model(torch.Tensor(Z_mean)).detach().numpy() * stdval2
        maxnorm = LA.norm(true, axis=1)
        recon_error = LA.norm(pred_mean + pred_mean2 - true, axis=1) / maxnorm * 100
        
        if idx in train_indices:
            train_errors_all.extend(recon_error)
        else:
            test_errors_all.extend(recon_error)
    
    results['kappa'].append(kappa)
    results['train_min'].append(np.min(train_errors_all))
    results['train_max'].append(np.max(train_errors_all))
    results['test_min'].append(np.min(test_errors_all))
    results['test_max'].append(np.max(test_errors_all))
    results['train_mean'].append(np.mean(train_errors_all))
    results['test_mean'].append(np.mean(test_errors_all))
    

# %% Plot min/max error vs kappa

fig, ax = plt.subplots(figsize=(8, 6))

kappas = np.array(results['kappa'])
train_min = np.array(results['train_min'])
train_max = np.array(results['train_max'])
test_min = np.array(results['test_min'])
test_max = np.array(results['test_max'])

train_mean = np.array(results['train_mean'])
test_mean = np.array(results['test_mean'])

# Plot second stage results with shaded regions
ax.fill_between(kappas, train_min, train_max, alpha=0.3, color='blue', label='Second Stage Train (min-max)')
ax.plot(kappas, train_min, 'b-', linewidth=2)
ax.plot(kappas, train_max, 'b-', linewidth=2)

ax.fill_between(kappas, test_min, test_max, alpha=0.3, color='red', label='Second Stage Test (min-max)')
ax.plot(kappas, test_min, 'r-', linewidth=2)
ax.plot(kappas, test_max, 'r-', linewidth=2)

# Plot first stage as horizontal lines for reference
ax.axhline(y=fs_train_min, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=fs_train_max, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='First Stage Train (min-max)')
ax.axhline(y=fs_test_min, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=fs_test_max, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='First Stage Test (min-max)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\\kappa$')
ax.set_ylabel('Relative Error (\\%)')
ax.set_title('Effect of $\\kappa$ on Prediction Error')
ax.legend(loc='best', fontsize=10)
ax.grid(True, which='both', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(f'results/kappa_error_range_{date_str}.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %% Alternative: separate min and max subplots

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # Min error plot
# ax = axes[0]
# ax.plot(kappas, train_min, 'bo-', linewidth=2, markersize=8, label='Second Stage Train')
# ax.plot(kappas, test_min, 'rs-', linewidth=2, markersize=8, label='Second Stage Test')
# ax.axhline(y=fs_train_min, color='blue', linestyle='--', linewidth=1.5, label='First Stage Train')
# ax.axhline(y=fs_test_min, color='red', linestyle='--', linewidth=1.5, label='First Stage Test')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel('$\\kappa$')
# ax.set_ylabel('Minimum Relative Error (\\%)')
# ax.set_title('Minimum Error vs $\\kappa$')
# ax.legend(loc='best', fontsize=9)
# ax.grid(True, which='both', linestyle=':', alpha=0.5)

# # Max error plot
# ax = axes[1]
# ax.plot(kappas, train_max, 'bo-', linewidth=2, markersize=8, label='Second Stage Train')
# ax.plot(kappas, test_max, 'rs-', linewidth=2, markersize=8, label='Second Stage Test')
# ax.axhline(y=fs_train_max, color='blue', linestyle='--', linewidth=1.5, label='First Stage Train')
# ax.axhline(y=fs_test_max, color='red', linestyle='--', linewidth=1.5, label='First Stage Test')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel('$\\kappa$')
# ax.set_ylabel('Maximum Relative Error (\\%)')
# ax.set_title('Maximum Error vs $\\kappa$')
# ax.legend(loc='best', fontsize=9)
# ax.grid(True, which='both', linestyle=':', alpha=0.5)

# plt.tight_layout()
# plt.savefig(f'results/kappa_minmax_separate_{date_str}.pdf', bbox_inches='tight', dpi=300)
# plt.show()

fig, axes = plt.subplots(1, 3,sharey=True, figsize=(18, 5))

# Min error plot
ax = axes[0]
ax.plot(kappas, train_min, 'bo-', linewidth=2, markersize=8, label='Second Stage Train')
ax.plot(kappas, test_min, 'rs-', linewidth=2, markersize=8, label='Second Stage Test')
ax.axhline(y=fs_train_min, color='blue', linestyle='--', linewidth=1.5, label='First Stage Train')
ax.axhline(y=fs_test_min, color='red', linestyle='--', linewidth=1.5, label='First Stage Test')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\\kappa$')
ax.set_ylabel('Relative Error (\\%)')
ax.set_title('Minimum Error vs $\\kappa$')
#ax.legend(loc='best', )
ax.grid(True, which='both', linestyle=':', alpha=0.5)

# Mean error plot
ax = axes[1]
ax.plot(kappas, train_mean, 'bo-', linewidth=2, markersize=8, label='Second Stage Train')
ax.plot(kappas, test_mean, 'rs-', linewidth=2, markersize=8, label='Second Stage Test')
ax.axhline(y=fs_train_mean, color='blue', linestyle='--', linewidth=1.5, label='First Stage Train')
ax.axhline(y=fs_test_mean, color='red', linestyle='--', linewidth=1.5, label='First Stage Test')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\\kappa$')
#ax.set_ylabel('Maximum Relative Error (\\%)')
ax.set_title('Mean Error vs $\\kappa$')
#ax.legend(loc='best', fontsize=9)
ax.grid(True, which='both', linestyle=':', alpha=0.5)

# Max error plot
ax = axes[2]
ax.plot(kappas, train_max, 'bo-', linewidth=2, markersize=8, label='Second Stage Train')
ax.plot(kappas, test_max, 'rs-', linewidth=2, markersize=8, label='Second Stage Test')
ax.axhline(y=fs_train_max, color='blue', linestyle='--', linewidth=1.5, label='First Stage Train')
ax.axhline(y=fs_test_max, color='red', linestyle='--', linewidth=1.5, label='First Stage Test')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\\kappa$')
#ax.set_ylabel('Maximum Relative Error (\\%)')
ax.set_title('Maximum Error vs $\\kappa$')
ax.legend(loc='best', )
ax.grid(True, which='both', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(f'results/kappa_minmax_separate_{date_str}.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %% Print summary table
print("\n" + "="*70)
print("Summary of Errors vs Kappa")
print("="*70)
print(f"{'Kappa':<10} {'Train Min':<12} {'Train Max':<12} {'Test Min':<12} {'Test Max':<12}")
print("-"*70)
print(f"{'1st Stage':<10} {fs_train_min:<12.4f} {fs_train_max:<12.4f} {fs_test_min:<12.4f} {fs_test_max:<12.4f}")
print("-"*70)
for i, kappa in enumerate(kappas):
    print(f"{kappa:<10} {train_min[i]:<12.4f} {train_max[i]:<12.4f} {test_min[i]:<12.4f} {test_max[i]:<12.4f}")
print("="*70)