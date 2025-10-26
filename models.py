import torch
from torch import nn
import numpy as np
from typing import Tuple
import torch.nn.functional as F

"""
    Complex-Weighted Convolutional Network (CWCN).

    - Uses a "complex weight learner" that learns real and imaginary parts of edge weights.
    - Performs diffusion on node features with these complex weights.
    - Supports left and right linear weight transformations per layer.
    - Uses epsilon parameters after each layer.
"""
class ComplexWeightsDiffusion(nn.Module):
    """
    Complex-Weighted Convolutional Network (CWCN).

    - Uses a "complex weight learner" that learns real and imaginary parts of edge weights.
    - Performs diffusion on node features with these complex weights.
    - Supports left and right linear weight transformations per layer.
    - Uses epsilon parameters after each layer.
    """
    def __init__(self,edge_index, args):
        super(ComplexWeightsDiffusion, self).__init__()
        self.edge_index = edge_index
        self.hidden_dim = args['hidden_channels'] * 2
        self.device = args['device']
        self.graph_size = args['graph_size']
        self.layers = args['layers']
        self.input_dropout = args['input_dropout']
        self.dropout = args['dropout']
        self.left_weights = args['left_weights']
        self.right_weights = args['right_weights']
        self.use_act = args['use_act']
        self.input_dim = args['input_dim']
        self.hidden_channels = args['hidden_channels']
        self.output_dim = args['output_dim']
        self.layers = args['layers']
        self.complex_weights_act = args['complex_weights_act'] 
        
        # Optional per-layer learnable weight matrices       
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        
        if self.right_weights:
            # Right weight matrices: act on hidden channels
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)#orth initializ
        if self.left_weights:
            # Left weight matrices: multiply by complex number (2x2)
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(2, 2, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)#initializ to identity
        
        # Module that learns a hermitian complex-weighted matrix (one complex weight per edge)
        self.complex_weights_learner=ComplexWeightsLearner(self.hidden_dim, out_shape=(2,),complex_weights_act=self.complex_weights_act)
        # Build sparse complex weight matrices from learned values
        self.weights_builder = WeightsBuilder(self.graph_size, edge_index)
        
        # Input transformation: feature dimension -> hidden_dim (with or without batch normalization)
        if not args["batch_norm"]:
            self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        else:
            self.lin1 =  nn.Sequential(
                        nn.Linear(self.input_dim, self.hidden_dim, bias=True),
                        nn.BatchNorm1d(self.hidden_dim)
                    )

        # Final output layer: hidden_dim -> num_classes
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Trainable epsilon parameters
        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((2, 1))))
    
    
    def forward(self, x):
        #Input projection: transform raw features into a feature matrix of size 2 x hidden_channels  
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        #Learn complex weight matrix
        x_maps = F.dropout(x, p=self.dropout, training=self.training)
        maps = self.complex_weights_learner(x_maps, self.edge_index)

        #Construct the weight matrices from the complex weights         
        W_real,W_imag=self.weights_builder(maps)
        #Modulus of the weight matrix 
        W_mod_2 = torch.sparse.sum(W_real.pow(2) + W_imag.pow(2), dim=1).to_dense()
        W_mod = torch.sqrt(W_mod_2 + 1e-6)
        #Inverse of the degree matrix         
        D_inv = 1.0 / W_mod           
        
        #Initial hidden state
        x0=x.view(self.graph_size * 2, -1)
        
        #Diffusion layers
        for layer in range(self.layers):

            x = F.dropout(x, p=self.dropout, training=self.training)
            
            #W_1 weight matrix 
            if self.left_weights:
                x = x.t().reshape(-1, 2)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * 2).t()

            #W_2 weight matrix
            if self.right_weights:
                x = self.lin_right_weights[layer](x)
            
            # Reshape into complex components                
            x = x.view(-1, 2, x.size(1))

            x_real=x[:, 0, :]
            x_imag=x[:, 1, :]

            # Diffusion step with complex multiplication     
            Wx_real=torch.sparse.mm(W_real, x_real)- torch.sparse.mm(W_imag, x_imag)
            Wx_imag=torch.sparse.mm(W_real, x_imag)+torch.sparse.mm(W_imag, x_real)

            p_x_real=Wx_real * D_inv.unsqueeze(1)
            p_x_imag=Wx_imag * D_inv.unsqueeze(1)
            
            x_real=x_real-p_x_real
            x_imag=x_imag-p_x_imag

            # Merge real and imag parts
            x=torch.stack([x_real,x_imag], dim=1).reshape(self.graph_size*2, -1)
            
            #Activation after diffusion layer
            if self.use_act:
                x = F.elu(x)
                
            #Epsilons 
            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))

            x0 = coeff * x0 - x
            x = x0
        
        #Output projection    
        x=x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
    
        
    def grouped_parameters(self):
        """
        Separate parameters into two groups:
        - Complex weight learner params (different weight decay)
        - All other params
        """
        weights_learners, others = [], []
        for name, param in self.named_parameters():
            if "weights_learner" in name:
                weights_learners.append(param)
            else:
                others.append(param)
        assert len(weights_learners) > 0
        assert len(weights_learners) + len(others) == len(list(self.parameters()))
        return weights_learners, others
        

class ComplexWeightsLearner(nn.Module):
    """
    Learns a complex weight for each edge, obtaining a Hermitian complex weights matrix.
    - For each edge (u, v), concatenates features of nodes u and v.
    - Applies linear transformation and activation.
    - Outputs 2D vector: [real_part, imag_part].
    """

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], complex_weights_act="tanh"):
        super(ComplexWeightsLearner, self).__init__()
        #Complex weight matrix
        self.W=None
        
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        
        # Linear projection from concatenated features (2*in_channels)
        self.linear1 = torch.nn.Linear(in_channels*2, int(np.prod(out_shape)), bias=False)
        
        # Select activation function for complex weights learning
        if complex_weights_act == 'id':
            self.act = lambda x: x
        elif complex_weights_act == 'tanh':
            self.act = torch.tanh
        elif complex_weights_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {complex_weights_act}")
        
    def forward(self,x,edge_index):
        row, col = edge_index #edge endpoints
        #Select node features of each edge 
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        
        # Learn weights from concatenated node features
        maps = self.linear1(torch.cat([x_row, x_col], dim=1))
        maps = self.act(maps)
        return maps

class WeightsBuilder(nn.Module):
    """
    Builds sparse complex weight matrices (real and imaginary parts) from edge weights.

    - Input: maps (edge weights with real and imag parts)
    - Output: W_real, W_imag (real and imaginary parts of the corresponding Hermitian complex weight matrix)
    """
    
    def __init__(self, size, edge_index):
        super(WeightsBuilder, self).__init__()
        self.size = size
        self.edges = edge_index.size(1) // 2
        self.edge_index = edge_index
        self.device = edge_index.device
        #Real and imaginary parts of the weight matrix
        self.W_real= None
        self.W_imag=None
        row, col = self.edge_index
        # COO index tensor for sparse matrix
        self.indices = torch.stack([row, col], dim=0).to(edge_index.device)
        
    def forward(self,maps):
        
        values_real, values_imag = maps[:, 0], maps[:, 1]
        # Build sparse weight matricex (real and imaginary parts of W)
        W_real = torch.sparse_coo_tensor(self.indices, values_real, (self.size, self.size))
        W_imag = torch.sparse_coo_tensor(self.indices, values_imag, (self.size, self.size))
        
        # Enforce antisymmetry
        W_real = (W_real + W_real.t()).coalesce() 
        W_imag = (W_imag - W_imag.t()).coalesce()  
    
        self.W_real= W_real.coalesce()
        self.W_imag=W_imag.coalesce()
        
        return W_real.coalesce(),W_imag.coalesce()
        


        