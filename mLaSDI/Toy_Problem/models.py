import torch
import torch.nn as nn
import math

class Autoencoder(torch.nn.Module):
    def __init__(self, space_dim, hidden_units, n_z):
        super(Autoencoder, self).__init__()

        n_layers = len(hidden_units)
        self.n_layers = n_layers
        

        fc1_e = torch.nn.Linear(space_dim, hidden_units[0])
        self.fc1_e = fc1_e

        if n_layers > 1:
            for i in range(n_layers - 1):
                fc_e = torch.nn.Linear(hidden_units[i], hidden_units[i + 1])
                setattr(self, 'fc' + str(i + 2) + '_e', fc_e)

        fc_e = torch.nn.Linear(hidden_units[-1], n_z)
        setattr(self, 'fc' + str(n_layers + 1) + '_e', fc_e)

        # g_e = torch.nn.Tanh()
        g_e = torch.nn.Softplus()

 
        self.g_e = g_e

        fc1_d = torch.nn.Linear(n_z, hidden_units[-1])
        self.fc1_d = fc1_d

        if n_layers > 1:
            for i in range(n_layers - 1, 0, -1):
                fc_d = torch.nn.Linear(hidden_units[i], hidden_units[i - 1])
                setattr(self, 'fc' + str(n_layers - i + 1) + '_d', fc_d)

        fc_d = torch.nn.Linear(hidden_units[0], space_dim)
        setattr(self, 'fc' + str(n_layers + 1) + '_d', fc_d)
        
        self.n_z = n_z



    def encoder(self, x):

        for i in range(1, self.n_layers + 1):
            fc = getattr(self, 'fc' + str(i) + '_e')
            x = self.g_e(fc(x))

        fc = getattr(self, 'fc' + str(self.n_layers + 1) + '_e')
        x = fc(x)


        return x


    def decoder(self, x):

        for i in range(1, self.n_layers + 1):
            fc = getattr(self, 'fc' + str(i) + '_d')
            x = self.g_e(fc(x))

        fc = getattr(self, 'fc' + str(self.n_layers + 1) + '_d')
        x = fc(x)

        return x


    def forward(self, x):

        x = Autoencoder.encoder(self, x)
        x = Autoencoder.decoder(self, x)

        return x
    

class CorrectingDecoder(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim, kappa=1.0):
        super(CorrectingDecoder, self).__init__()
        self.kappa = kappa

        # Store layers as a ModuleList to access them individually
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        # First hidden layer with sine activation
        if hidden_units:
            self.layers.append(nn.Linear(prev_dim, hidden_units[0]))
            prev_dim = hidden_units[0]

            # Remaining hidden layers
            for hidden_dim in hidden_units[1:]:
                self.layers.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim

        # Output layer (linear)
        self.layers.append(nn.Linear(prev_dim, output_dim))

        # Initialize weights using Xavier initialization from Wang and Lai
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization with truncated normal"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                in_dim = layer.weight.size(1)
                out_dim = layer.weight.size(0)
                # Xavier initialization
                xavier_stddev = math.sqrt(2.0 / (in_dim + out_dim))

                # Initialize with truncated normal (clamp to 2 standard deviations)
                with torch.no_grad():
                    layer.weight.normal_(0, xavier_stddev)
                    layer.weight.clamp_(-2*xavier_stddev, 2*xavier_stddev)

                    # Initialize biases to zero
                    layer.bias.zero_()

    def forward(self, x):
        # Apply first layer with kappa scaling (matching Wang and Lai)
        if len(self.layers) > 0:
            x = self.kappa * self.layers[0](x)
            x = torch.sin(x)  # Sine activation for first layer

            # Remaining hidden layers with tanh activation
            for layer in self.layers[1:-1]:
                x = torch.tanh(layer(x))
                # x = torch.nn.functional.softplus(layer(x))

            # Final output layer (no activation)
            if len(self.layers) > 1:
                x = self.layers[-1](x)
            
        
        return x
  