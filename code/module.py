"""
Neural Network Modules Implementation

This module provides a complete implementation of neural network building blocks
from scratch, including:
- Basic layers: Linear, Embedding, Sequential
- Activation functions: ReLU, Tanh, Softmax
- Normalization layers: BatchNorm1d, LayerNorm
- Regularization: Dropout
- Recurrent layers: RNN, LSTM, GRU
- Utility layers: Flatten

All modules are designed to be compatible with PyTorch tensors and follow
similar interfaces to PyTorch's nn.Module.

Author: Nazar Korniichuk
Date: 02.07.2025
"""

import torch

#Sequential model
class Sequential:
    """
    Sequential container for stacking multiple layers.
    
    Similar to PyTorch's nn.Sequential, this allows easy composition of
    multiple layers into a single callable module.
    
    Args:
        layers (list): List of layer objects to stack
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, layers, device = 'cpu'):
        self.device = device
        self.layers = layers
    
    def to(self, device):
        """Move all layers to the specified device."""
        self.device = device
        for layer in self.layers:
            layer.to(device)
        return self
    
    def __call__(self, x):
        """Forward pass through all layers sequentially."""
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        """Return all trainable parameters from all layers."""
        return [p for layer in self.layers for p in layer.parameters()]

    def append(self, layer):
        """Add a new layer to the end of the sequence."""
        self.layers.append(layer)

#Embedding layer
class Embedding:
    """
    Embedding layer for learning dense representations of discrete inputs.
    
    Maps integer indices to dense vectors of fixed size. Commonly used
    for word embeddings, character embeddings, etc.
    
    Args:
        num_embeddings (int): Size of the vocabulary
        embedding_dim (int): Size of each embedding vector
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, num_embeddings, embedding_dim, device = 'cpu'):
        self.device = device
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.randn(num_embeddings, embedding_dim, device = device)
    
    def to(self, device):
        """Move embedding weights to the specified device."""
        self.device = device
        self.weight = self.weight.to(device)
        return self

    def __call__(self, x):
        self.out = self.weight[x]
        return self.out
    
    def parameters(self):
        """Return the embedding weight tensor."""
        return [self.weight]

#Linear layer
class Linear:
    """
    Linear (fully connected) layer.
    
    Implements y = xW + b where W is the weight matrix and b is the bias vector.
    Uses Kaiming initialization for better training stability.
    
    Args:
        fan_in (int): Number of input features
        fan_out (int): Number of output features
        bias (bool): Whether to include bias term (default: True)
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, fan_in, fan_out, bias=True, device = 'cpu'):
        self.device = device
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.weight = torch.randn(fan_in, fan_out, device = device)/fan_in**0.5 #kaiming initialization
        self.bias = torch.zeros(fan_out, device = device) if bias else None
    
    def to(self, device):
        """Move weights and bias to the specified device."""
        self.device = device
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device) if self.bias is not None else None
        return self

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        """Return weight and bias parameters."""
        return [self.weight] + ([] if self.bias is None else [self.bias])

#Tanh activation function
class Tanh:
    """
    Hyperbolic tangent activation function.
    
    Outputs values in the range [-1, 1]. Useful for bounded outputs
    and as an alternative to sigmoid in some cases.
    
    Args:
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, device = 'cpu'):
        self.device = device
    def __call__(self, x):
        """Apply tanh activation: f(x) = tanh(x)."""
        self.out = torch.tanh(x)
        return self.out
    def to(self, device):
        """Move to the specified device."""
        self.device = device
        return self
    def parameters(self):
        """No trainable parameters."""
        return []

#softmax activation function
class Softmax:
    """
    Softmax activation function.
    
    Converts input logits to probability distribution. Commonly used
    as the final layer in classification tasks.
    
    Args:
        dim (int): Dimension along which to apply softmax (default: 1)
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, dim = 1, device = 'cpu'):
        self.dim = dim
        self.device = device
    def __call__(self, x):
        """Apply softmax activation: f(x_i) = exp(x_i) / sum(exp(x_j))."""
        self.out = torch.softmax(x, dim=self.dim)
        return self.out
    def to(self, device):
        """Move to the specified device."""
        self.device = device
        return self
    def parameters(self):
        """No trainable parameters."""
        return []

#ReLU activation function
class ReLU:
    """
    Rectified Linear Unit activation function.
    
    Implements f(x) = max(0, x). Most commonly used activation function
    in modern neural networks due to its simplicity and effectiveness.
    
    Args:
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, device = 'cpu'):
        self.device = device
    def __call__(self, x):
        """Apply ReLU activation: f(x) = max(0, x)."""
        self.out = torch.relu(x)
        return self.out
    def to(self, device):
        """Move to the specified device."""
        self.device = device
        return self
    def parameters(self):
        """No trainable parameters."""
        return []

#Flatten layer
class Flatten:
    """
    Flatten layer for reshaping tensors.
    
    Flattens specified dimensions of the input tensor. Useful for
    transitioning from convolutional layers to fully connected layers.
    
    Args:
        start_dim (int): First dimension to flatten (default: 1)
        end_dim (int): Last dimension to flatten (default: -1)
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, start_dim=1, end_dim=-1, device = 'cpu'):
        self.device = device
        self.start_dim = start_dim
        self.end_dim = end_dim
    def __call__(self, x):
        """
        Flatten the tensor along specified dimensions.
        
        Example: (batch, channels, height, width) -> (batch, channels*height*width)
        """
        shape = list()
        if self.end_dim == -1:
            for i, dim in enumerate(x.shape):
                if (i <= self.start_dim) or (len(shape) == 0):
                    shape.append(dim)
                else:
                    shape[-1] *= dim
        else:
            for i, dim in enumerate(x.shape):
                if (i <= self.start_dim) or (len(shape) == 0):
                    shape.append(dim)
                elif i > self.end_dim:
                    shape.append(dim)
                else:
                    shape[-1] *= dim
        self.out = x.view(shape)
        return self.out
    def to(self, device):
        """Move to the specified device."""
        self.device = device
        return self
    def parameters(self):
        """No trainable parameters."""
        return []

#Vanila RNN layer
class RNN:
    """
    Vanilla Recurrent Neural Network layer.
    
    Implements a simple RNN with tanh activation. Processes sequences
    by maintaining a hidden state that gets updated at each timestep.
    
    Args:
        n_input (int): Number of input features
        n_hidden (int): Number of hidden units
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, n_input, n_hidden, device = 'cpu'):
        self.device = device
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.W_xh = torch.randn(n_input, n_hidden, device = device) / n_input**0.5 #kaiming initialization
        self.W_hh = torch.randn(n_hidden, n_hidden, device = device) / n_hidden**0.5 #kaiming initialization
        self.b_h = torch.zeros(n_hidden, device = device)
        
    def to(self, device):
        """Move all parameters to the specified device."""
        self.device = device
        self.W_xh = self.W_xh.to(device)
        self.W_hh = self.W_hh.to(device)
        self.b_h = self.b_h.to(device)
        return self
    
    def __call__(self, x):
        """
        Forward pass through the RNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_input)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_hidden)
        """
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.n_hidden).to(self.device)
        output = []
        for i in range(seq_len):
            h_t = torch.tanh(x[:, i, :] @ self.W_xh + h_t @ self.W_hh + self.b_h)
            output.append(h_t)

        self.h = torch.stack(output, dim=1)
        return self.h
    def parameters(self):
        """Return all trainable parameters."""
        return [self.W_xh, self.W_hh, self.b_h]

#LSTM layer
class LSTM:
    """
    Long Short-Term Memory (LSTM) layer.
    
    Implements LSTM with input, forget, cell, and output gates.
    Better at capturing long-term dependencies than vanilla RNN.
    
    Args:
        n_input (int): Number of input features
        n_hidden (int): Number of hidden units
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, n_input, n_hidden, device = 'cpu'):
        self.device = device
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.W_ii = torch.randn(n_input, n_hidden, device = device) / n_input**0.5 #kaiming initialization
        self.W_hi = torch.randn(n_hidden, n_hidden, device = device) / n_hidden**0.5 #kaiming initialization
        self.b_i = torch.zeros(n_hidden, device = device)
        
        self.W_if = torch.randn(n_input, n_hidden, device = device) / n_input**0.5 #kaiming initialization
        self.W_hf = torch.randn(n_hidden, n_hidden, device = device) / n_hidden**0.5 #kaiming initialization
        self.b_f = torch.zeros(n_hidden, device = device)
        
        self.W_ig = torch.randn(n_input, n_hidden, device = device) / n_input**0.5 #kaiming initialization
        self.W_hg = torch.randn(n_hidden, n_hidden, device = device) / n_hidden**0.5 #kaiming initialization
        self.b_g = torch.zeros(n_hidden, device = device)
        
        self.W_io = torch.randn(n_input, n_hidden, device = device) / n_input**0.5 #kaiming initialization
        self.W_ho = torch.randn(n_hidden, n_hidden, device = device) / n_hidden**0.5 #kaiming initialization
        self.b_o = torch.zeros(n_hidden, device = device)
    
    def to(self, device):
        """Move all parameters to the specified device."""
        self.device = device
        self.W_ii = self.W_ii.to(device)
        self.W_hi = self.W_hi.to(device)
        self.b_i = self.b_i.to(device)
        self.W_if = self.W_if.to(device)
        self.W_hf = self.W_hf.to(device)
        self.b_f = self.b_f.to(device)
        self.W_ig = self.W_ig.to(device)
        self.W_hg = self.W_hg.to(device)
        self.b_g = self.b_g.to(device)
        self.W_io = self.W_io.to(device)
        self.W_ho = self.W_ho.to(device)
        self.b_o = self.b_o.to(device)
        return self

        
    def __call__(self, x):
        """
        Forward pass through the LSTM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_input)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_hidden)
        """
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.n_hidden).to(self.device)
        c_t = torch.zeros(batch_size, self.n_hidden).to(self.device)
        output = []
        for i in range(seq_len):
            i_t = torch.sigmoid(x[:, i, :] @ self.W_ii + h_t @ self.W_hi + self.b_i)
            f_t = torch.sigmoid(x[:, i, :] @ self.W_if + h_t @ self.W_hf + self.b_f)
            g_t = torch.tanh(x[:, i, :] @ self.W_ig + h_t @ self.W_hg + self.b_g)
            o_t = torch.sigmoid(x[:, i, :] @ self.W_io + h_t @ self.W_ho + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            output.append(h_t)
        self.h = torch.stack(output, dim=1)
        self.c = c_t
        return self.h
    def parameters(self):
        """Return all trainable parameters."""
        return [self.W_ii, self.W_hi, self.b_i, self.W_if, self.W_hf, self.b_f, self.W_ig, self.W_hg, self.b_g, self.W_io, self.W_ho, self.b_o]

#GRU layer
class GRU:
    """
    Gated Recurrent Unit (GRU) layer.
    
    Implements GRU with reset and update gates. Simpler than LSTM but
    often performs similarly well on many tasks.
    
    Args:
        n_input (int): Number of input features
        n_hidden (int): Number of hidden units
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, n_input, n_hidden, device = 'cpu'):
        self.device = device
        self.n_input = n_input
        self.n_hidden = n_hidden
        
        self.W_ir = torch.randn(n_input, n_hidden, device = device) / n_input**0.5 #kaiming initialization
        self.W_hr = torch.randn(n_hidden, n_hidden, device = device) / n_hidden**0.5 #kaiming initialization
        self.b_r = torch.zeros(n_hidden, device = device)
        
        self.W_iz = torch.randn(n_input, n_hidden, device = device) / n_input**0.5 #kaiming initialization
        self.W_hz = torch.randn(n_hidden, n_hidden, device = device) / n_hidden**0.5 #kaiming initialization
        self.b_z = torch.zeros(n_hidden, device = device)
        
        self.W_in = torch.randn(n_input, n_hidden, device = device) / n_input**0.5 #kaiming initialization
        self.W_hn = torch.randn(n_hidden, n_hidden, device = device) / n_hidden**0.5 #kaiming initialization
        self.b_in = torch.zeros(n_hidden, device = device)
        self.b_hn = torch.zeros(n_hidden, device = device)
        
    def to(self, device):
        """Move all parameters to the specified device."""
        self.device = device
        self.W_ir = self.W_ir.to(device)
        self.W_hr = self.W_hr.to(device)
        self.b_r = self.b_r.to(device)
        self.W_iz = self.W_iz.to(device)
        self.W_hz = self.W_hz.to(device)
        self.b_z = self.b_z.to(device)
        self.W_in = self.W_in.to(device)
        self.W_hn = self.W_hn.to(device)
        self.b_in = self.b_in.to(device)
        self.b_hn = self.b_hn.to(device)
        return self
    
    def parameters(self):
        """Return all trainable parameters."""
        return [self.W_ir, self.W_hr, self.b_r, self.W_iz, self.W_hz, self.b_z, self.W_in, self.W_hn, self.b_in, self.b_hn]
    
    def __call__(self, x):
        """
        Forward pass through the GRU.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_input)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, n_hidden)
        """
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.n_hidden).to(self.device)
        output = []
        
        for i in range(seq_len):
            r_t = torch.sigmoid(x[:, i, :] @ self.W_ir + h_t @ self.W_hr + self.b_r)
            z_t = torch.sigmoid(x[:, i, :] @ self.W_iz + h_t @ self.W_hz + self.b_z)
            n_t = torch.tanh(x[:, i, :] @ self.W_in + self.b_in + r_t * (h_t @ self.W_hn + self.b_hn))
            h_t = (1 - z_t) * n_t + z_t * h_t
            output.append(h_t)
        self.h = torch.stack(output, dim=1)
        
        return self.h

#Batch normalization
class BatchNorm1d:
    """
    1D Batch Normalization layer.
    
    Normalizes activations across the batch dimension and learns
    scale and shift parameters. Helps with training stability and speed.
    
    Args:
        num_features (int): Number of features to normalize
        eps (float): Small constant for numerical stability (default: 1e-5)
        momentum (float): Momentum for running statistics (default: 0.1)
        training (bool): Whether in training mode (default: True)
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, training=True, device = 'cpu'):
        self.device = device
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = training
        self.running_mean = torch.zeros(num_features, device = device)
        self.running_var = torch.ones(num_features, device = device)
        self.gamma = torch.ones(num_features, device = device)
        self.beta = torch.zeros(num_features, device = device)
    
    def to(self, device):
        """Move all parameters to the specified device."""
        self.device = device
        self.running_mean = self.running_mean.to(device)
        self.running_var = self.running_var.to(device)
        self.gamma = self.gamma.to(device)
        self.beta = self.beta.to(device)
        return self
        
    def __call__(self, x):
        """
        Forward pass of batch normalization.
        
        In training mode: normalize using batch statistics and update running stats
        In eval mode: normalize using running statistics
        """
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            x_mean = x.mean(dim=dim, keepdim=True)
            x_var = x.var(dim=dim, keepdim=True)
            with torch.no_grad():
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * x_mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * x_var
        else:
            x_mean = self.running_mean
            x_var = self.running_var
        x_std = torch.sqrt(x_var + self.eps)
        x_hat = (x - x_mean) / x_std
        self.out = x_hat * self.gamma + self.beta
        return self.out
    
    def parameters(self):
        """Return learnable parameters (gamma and beta)."""
        return [self.gamma, self.beta]

#Layer normalization
class LayerNorm:
    """
    Layer Normalization layer.
    
    Normalizes activations across the feature dimension for each sample.
    Unlike batch norm, this works well with variable batch sizes and
    is commonly used in transformers.
    
    Args:
        normalized_shape (int): Number of features to normalize
        eps (float): Small constant for numerical stability (default: 1e-5)
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, normalized_shape, eps=1e-5, device = 'cpu'):
        self.device = device
        self.num_features = normalized_shape
        self.eps = eps
        self.gamma = torch.ones(normalized_shape, device = device)
        self.beta = torch.zeros(normalized_shape, device = device)
        
    def __call__(self, x):
        """
        Forward pass of layer normalization.
        
        Normalizes across the last dimension for each sample independently.
        """
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True, unbiased=False)
        self.out = (x - x_mean) / torch.sqrt(x_var + self.eps) * self.gamma + self.beta
        return self.out
    
    def parameters(self):
        """Return learnable parameters (gamma and beta)."""
        return [self.gamma, self.beta]
    
    def to(self, device):
        """Move parameters to the specified device."""
        self.device = device
        self.gamma = self.gamma.to(device)
        self.beta = self.beta.to(device)
        return self

class Dropout:
    """
    Dropout layer for regularization.
    
    Randomly sets a fraction of inputs to zero during training to prevent
    overfitting. During evaluation, scales the output by (1 - p).
    
    Args:
        p (float): Probability of dropping an element (default: 0.5)
        training (bool): Whether in training mode (default: True)
        device (str): Device to place tensors on (default: 'cpu')
    """
    
    def __init__(self, p=0.5, training=True, device = 'cpu'):
        self.device = device
        self.p = p
        self.training = training
    def eval(self):
        """Set to evaluation mode (no dropout)."""
        self.training = False
        return self
    def train(self):
        """Set to training mode (apply dropout)."""
        self.training = True
        return self
        
    def __call__(self, x):
        """
        Forward pass of dropout.
        
        In training: randomly zero some elements and scale by 1/(1-p)
        In eval: return input unchanged
        """
        if self.training:
            self.out = x * (torch.rand_like(x, device=self.device) > self.p).float() / (1 - self.p)
        else:
            self.out = x
        return self.out
    def to(self, device):
        """Move to the specified device."""
        self.device = device
        return self
    def parameters(self):
        """No trainable parameters."""
        return []
