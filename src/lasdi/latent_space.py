import torch
import numpy as np
import operator
from functools import reduce
from torch import nn
from .fno_shared import *
import pdb
import time

# activation dict
act_dict = {'ELU': torch.nn.ELU,
            'hardshrink': torch.nn.Hardshrink,
            'hardsigmoid': torch.nn.Hardsigmoid,
            'hardtanh': torch.nn.Hardtanh,
            'hardswish': torch.nn.Hardswish,
            'leakyReLU': torch.nn.LeakyReLU,
            'logsigmoid': torch.nn.LogSigmoid,
            'multihead': torch.nn.MultiheadAttention,
            'PReLU': torch.nn.PReLU,
            'ReLU': torch.nn.ReLU,
            'ReLU6': torch.nn.ReLU6,
            'RReLU': torch.nn.RReLU,
            'SELU': torch.nn.SELU,
            'CELU': torch.nn.CELU,
            'gelu': torch.nn.GELU,
            'sigmoid': torch.nn.Sigmoid,
            'SiLU': torch.nn.SiLU,
            'mish': torch.nn.Mish,
            'softplus': torch.nn.Softplus,
            'softshrink': torch.nn.Softshrink,
            'tanh': torch.nn.Tanh,
            'tanhshrink': torch.nn.Tanhshrink,
            'threshold': torch.nn.Threshold,
            }

def initial_condition_latent(param_grid, physics, autoencoder):

    '''

    Outputs the initial condition in the latent space: Z0 = encoder(U0)

    '''

    n_param = param_grid.shape[0]
    Z0 = []

    sol_shape = [1, 1] + physics.qgrid_size

    for i in range(n_param):
        u0 = physics.initial_condition(param_grid[i])
        u0 = u0.reshape(sol_shape)
        u0 = torch.Tensor(u0)
        z0 = autoencoder.encoder(u0)
        z0 = z0[0, 0, :].detach().numpy()
        Z0.append(z0)

    return Z0

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, layer_sizes,
                 act_type='sigmoid', reshape_index=None, reshape_shape=None,
                 threshold=0.1, value=0.0, num_heads=1):
        super(MultiLayerPerceptron, self).__init__()

        # including input, hidden, output layers
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        # Linear features between layers
        self.fcs = []
        for k in range(self.n_layers-1):
            self.fcs += [torch.nn.Linear(layer_sizes[k], layer_sizes[k + 1])]
        self.fcs = torch.nn.ModuleList(self.fcs)
        self.init_weight()

        # Reshape input or output layer
        assert((reshape_index is None) or (reshape_index in [0, -1]))
        assert((reshape_shape is None) or (np.prod(reshape_shape) == layer_sizes[reshape_index]))
        self.reshape_index = reshape_index
        self.reshape_shape = reshape_shape

        # Initalize activation function
        self.act_type = act_type
        self.use_multihead = False
        if act_type == "threshold":
            self.act = act_dict[act_type](threshold, value)

        elif act_type == "multihead":
            self.use_multihead = True
            if (self.n_layers > 3): # if you have more than one hidden layer
                self.act = []
                for i in range(self.n_layers-2):
                    self.act += [act_dict[act_type](layer_sizes[i+1], num_heads)]
            else:
                self.act = [torch.nn.Identity()]  # No additional activation
            self.act = torch.nn.ModuleList(self.fcs)

        #all other activation functions initialized here
        else:
            self.act = act_dict[act_type]()
        return
    
    def forward(self, x):
        if (self.reshape_index == 0):
            # make sure the input has a proper shape
            assert(list(x.shape[-len(self.reshape_shape):]) == self.reshape_shape)
            # we use torch.Tensor.view instead of torch.Tensor.reshape,
            # in order to avoid data copying.
            x = x.view(list(x.shape[:-len(self.reshape_shape)]) + [self.layer_sizes[self.reshape_index]])

        for i in range(self.n_layers-2):
            x = self.fcs[i](x) # apply linear layer
            if (self.use_multihead):
                x = self.apply_attention(self, x, i)
            else:
                x = self.act(x)

        x = self.fcs[-1](x)

        if (self.reshape_index == -1):
            # we use torch.Tensor.view instead of torch.Tensor.reshape,
            # in order to avoid data copying.
            x = x.view(list(x.shape[:-1]) + self.reshape_shape)

        return x
    
    def apply_attention(self, x, act_idx):
        x = x.unsqueeze(1)  # Add sequence dimension for attention
        x, _ = self.act[act_idx](x, x, x) # apply attention
        x = x.squeeze(1)  # Remove sequence dimension
        return x
    
    def init_weight(self):
        # TODO(kevin): support other initializations?
        for fc in self.fcs:
            torch.nn.init.xavier_uniform_(fc.weight)
        return

class Autoencoder(torch.nn.Module):

    def __init__(self, physics, config):
        super(Autoencoder, self).__init__()

        self.qgrid_size = physics.qgrid_size
        self.space_dim = np.prod(self.qgrid_size)
        hidden_units = config['hidden_units']
        n_z = config['latent_dimension']
        self.n_z = n_z

        layer_sizes = [self.space_dim] + hidden_units + [n_z]
        #grab relevant initialization values from config
        act_type = config['activation'] if 'activation' in config else 'sigmoid'
        threshold = config["threshold"] if "threshold" in config else 0.1
        value = config["value"] if "value" in config else 0.0
        num_heads = config['num_heads'] if 'num_heads' in config else 1

        self.encoder = MultiLayerPerceptron(layer_sizes, act_type,
                                            reshape_index=0, reshape_shape=self.qgrid_size,
                                            threshold=threshold, value=value, num_heads=num_heads)
        
        self.decoder = MultiLayerPerceptron(layer_sizes[::-1], act_type,
                                            reshape_index=-1, reshape_shape=self.qgrid_size,
                                            threshold=threshold, value=value, num_heads=num_heads)
        
        # Print number of parameters
        n_params = self.count_params()
        print(f'Autoencoder has {n_params} parameters.')

        return

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
    def export(self):
        dict_ = {'autoencoder_param': self.cpu().state_dict()}
        return dict_
    
    def load(self, dict_):
        self.load_state_dict(dict_['autoencoder_param'])
        return
    
    def count_params(self):
        n_params = 0
        for param in self.parameters():
            n_params += param.numel()
        return n_params
    

class FNO_Autoencoder_1d(torch.nn.Module):

    """
    Fourier Neural Operator for mapping functions to functions
    
        modes1          (int): Fourier mode truncation levels
        width           (int): constant dimension of channel space
        s_outputspace   (int): desired spatial resolution (s,) in output space
        width_final     (int): width of the final projection layer
        padding         (int or float): (1.0/padding) is fraction of domain to zero pad (non-periodic)
        d_in            (int): number of input channels (NOT including grid input features)
        d_out           (int): number of output channels (co-domain dimension of output space functions)
        act             (str): Activation function = tanh, relu, gelu, elu, or leakyrelu
        n_layers        (int): Number of Fourier Layers, by default 4
        get_grid        (bool): Whether or not append grid coordinate as a feature for the input
    """
    def __init__(self, physics, config):

        super(FNO_Autoencoder_1d, self).__init__()

        self.d_physical = config['d_physical']
        self.modes1 = config['modes']
        self.layer_widths = config['layer_widths']
        self.n_z= config['hidden_dim'] 
        self.padding = 8 if 'padding' not in config else config['padding']
        self.d_in= config['d_in']
        self.pointwise_lift_dim = config['pointwise_lift_dim']
        self.act = act_dict[config['activation'] if 'activation' in config else 'gelu']()
        self.n_layers = len(self.layer_widths)
        self.get_grid = True
        self.n_samples = None
        self.adjusted = False # Flag that encoder adjusted dimensions
        
        
        self.set_outputspace_resolution(physics.nt)

        ##### ENCODER #####
        self.fc0_e = torch.nn.Linear((self.d_in + self.d_physical) if self.get_grid else self.d_in, self.pointwise_lift_dim)

        
        # self.speconvs_e = nn.ModuleList([SpectralConv1d(self.layer_widths[i],self.layer_widths[i+1], self.modes1)
        #                                for i in range(self.n_layers-1)])

        # self.ws_e = nn.ModuleList([
        #     nn.Conv1d(self.pointwise_lift_dim, self.pointwise_lift_dim, 1)
        #         for i in range(self.n_layers-1)]
        #     )

        self.lfunc0_e = LinearFunctionals1d(self.pointwise_lift_dim, self.layer_widths[-1], self.modes1)
        self.mlpfunc0_e = MLP(self.pointwise_lift_dim, self.pointwise_lift_dim, self.pointwise_lift_dim, 'gelu')

        # self.mlpfunc0_e = MLP(self.pointwise_lift_dim, self.pointwise_lift_dim, self.layer_widths[-1], 'gelu')
        # self.mlpfunc0_e = nn.Sequential( # replaces MLP
        #     nn.Conv1d(self.pointwise_lift_dim, self.pointwise_lift_dim, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv1d(self.pointwise_lift_dim, self.pointwise_lift_dim, kernel_size=1)
        # )

        self.mlp0_e = MLP(self.layer_widths[-1], self.layer_widths[-1], self.n_z, 'gelu')
        
        ##### DECODER #####

        self.mlp0_d = MLP(self.n_z, self.layer_widths[-1], self.layer_widths[-1], 'gelu')


        self.ldec0_d = LinearDecoder1d(self.layer_widths[-1],self.pointwise_lift_dim, self.modes1)
        
        # self.speconvs_d = nn.ModuleList([SpectralConv1d(self.pointwise_lift_dim, self.pointwise_lift_dim,self.modes1) 
        #                                 ])

        # self.ws_d = nn.ModuleList([
        #     nn.Conv1d(self.pointwise_lift_dim, self.pointwise_lift_dim, 1)
        #         ]
        #     )
        # add one MLP layer
        self.fc1_d = MLP(self.pointwise_lift_dim, self.pointwise_lift_dim, self.d_in, 'gelu')

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
    def encoder(self, x):
        '''
        standard encoder expects data of the form (n_samples, n_t, n_channels, n_x)

        '''

        # if self.d_in == 1:
        #     # reshape so that n_samples and n_t are in the same dimension
        #     self.n_samples = x.shape[0]
        #     x = x.reshape(-1,self.d_in, x.shape[-1])
        #     self.adjusted = True

        # reshape so that n_samples and n_t are same dimension
        start_time = time.time()
        n_x = x.shape[-1]
        self.set_outputspace_resolution(n_x)

        self.n_samples = x.shape[0]
        x = x.reshape(-1,self.d_in, x.shape[-1]) # (batch, d_in, n_x)
        self.adjusted = True
        reshape_time = time.time()




        # # Lifting layer
        if self.get_grid:
            x = torch.cat((x, get_grid1d(x.shape, x.device)), dim=-2)    # grid ``features''
       
        grid_time = time.time()

        x = x.permute(0, 2, 1)  # (batch, n_x, d_in + d_physical)
        x = self.fc0_e(x) # Linear pointwise lift
        x = x.permute(0, 2, 1)  # (batch, pointwise_lift_dim, n_x)
        
        lift_layer_time = time.time()
        print("Lift layer time:", lift_layer_time - grid_time)
        # Map from input domain into the torus
        x = F.pad(x, [0, x.shape[-1]//self.padding])
        # # Fourier integral operator layers on the torus
        # for speconv, w in zip(self.speconvs_e, self.ws_e):
        #     x = w(x) + speconv(x)
        #     x = self.act(x)

        # Extract Fourier neural functionals on the torus
        x = self.lfunc0_e(x)
        nf_time = time.time()
        print("Neural functional layer time:", nf_time - lift_layer_time)
        
        # Retain the truncated modes (use all modes) (local information)
        # x = x.permute(0, 2, 1)
        # x = self.mlpfunc0_e(x)
        # x = x.permute(0, 2, 1)
        # x = x.mean(dim=-1 ) # used to be trapz assuming uniform 
        
        # Combine nonlocal and local features
        # x = torch.cat((x_temp, x), dim=1)


        # Final projection layer
        x = self.mlp0_e(x)

        proj_time = time.time()
        print("Final projection layer time:", proj_time - nf_time)

        if self.adjusted:
            x = x.reshape(self.n_samples, -1, x.shape[-1])
            self.adjusted = False

        return x
    
    def decoder(self, x):
    
        """
        Input shape (of x):     (batch, self.n_z)
        Output shape:           (batch, self.d_out, nx_out)
        
        The output resolution is determined by self.s_outputspace
        """

        self.n_samples = x.shape[0]
        x = x.reshape(-1, self.n_z)
        self.adjusted = True

            
        
        # Lifting layer
        x = self.mlp0_d(x)
        
        # Decode into functions on the torus
        x = self.ldec0_d(x, self.s_outputspace)
        
        # # Fourier integral operator layers on the torus
        # for idx_layer, (speconv, w) in enumerate(zip(self.speconvs_d, self.ws_d)):
        #     x = w(x) + speconv(x)
        #     if idx_layer != self.n_layers - 2:
        #         x = self.act(x)
           
        # Map from the torus into the output domain
        x = x[..., :-self.num_pad_outputspace]

        x = x.permute(0, 2, 1)
        x = self.fc1_d(x)
        x = x.permute(0, 2, 1)

        if self.adjusted:
            x = x.reshape(self.n_samples, -1, x.shape[-1])
            self.adjusted = False
           
        return x

    
    def export(self):
        dict_ = {'autoencoder_param': self.cpu().state_dict()}
        return dict_
    
    def load(self, dict_):
        self.load_state_dict(dict_['autoencoder_param'])
        return
    

    def set_outputspace_resolution(self, s=None):
        """
        Helper to set desired output space resolution of the model at any time
        """
        if s is None:
            self.s_outputspace = None
            self.num_pad_outputspace = None
        else:
            self.s_outputspace = s + s//self.padding
            self.num_pad_outputspace = s//self.padding

    def count_params(self):
        """
        print the number of parameters
        """
        c = 0
        for p in list(self.parameters()):
            c += reduce(operator.mul,
                        list(p.size()+(2,) if p.is_complex() else p.size()))
        return c
       