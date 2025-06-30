import torch

#Sequential model
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def append(self, layer):
        self.layers.append(layer)

#Embedding layer
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.randn(num_embeddings, embedding_dim)

    def __call__(self, x):
        self.out = self.weight[x]
        return self.out
    
    def parameters(self):
        return [self.weight]

#Linear layer
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.weight = torch.randn(fan_in, fan_out)/fan_in**0.5 #kaiming initialization
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

#Tanh activation function
class Tanh:
    def __init__(self):
        pass
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []

#softmax activation function
class Softmax:
    def __init__(self, dim = 1):
        self.dim = dim
    def __call__(self, x):
        self.out = torch.softmax(x, dim=self.dim)
        return self.out
    def parameters(self):
        return []

#ReLU activation function
class ReLU:
    def __init__(self):
        pass
    def __call__(self, x):
        self.out = torch.relu(x)
        return self.out
    def parameters(self):
        return []

#Flatten layer
class Flatten:
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim
    def __call__(self, x):
        #(d_1, d_2, ..., d_n) -> (d_1, d_2*...*d_)
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
    def parameters(self):
        return []

#Vanila RNN layer
class RNN:
    def __init__(self, n_input, n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.W_xh = torch.randn(n_input, n_hidden) / n_input**0.5 #kaiming initialization
        self.W_hh = torch.randn(n_hidden, n_hidden) / n_hidden**0.5 #kaiming initialization
        self.b_h = torch.zeros(n_hidden)
        
    def __call__(self, x):
        #(N, L, H_in) -> (N, L, H_out)
        #h_t -> (N, H_out)
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.n_hidden)
        output = []
        for i in range(seq_len):
            h_t = torch.tanh(x[:, i, :] @ self.W_xh + h_t @ self.W_hh + self.b_h)
            output.append(h_t)

        self.h = torch.stack(output, dim=1)
        return self.h
    def parameters(self):
        return [self.W_xh, self.W_hh, self.b_h]

#LSTM layer
class LSTM:
    def __init__(self, n_input, n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.W_ii = torch.randn(n_input, n_hidden) / n_input**0.5 #kaiming initialization
        self.W_hi = torch.randn(n_hidden, n_hidden) / n_hidden**0.5 #kaiming initialization
        self.b_i = torch.zeros(n_hidden)
        
        self.W_if = torch.randn(n_input, n_hidden) / n_input**0.5 #kaiming initialization
        self.W_hf = torch.randn(n_hidden, n_hidden) / n_hidden**0.5 #kaiming initialization
        self.b_f = torch.zeros(n_hidden)
        
        self.W_ig = torch.randn(n_input, n_hidden) / n_input**0.5 #kaiming initialization
        self.W_hg = torch.randn(n_hidden, n_hidden) / n_hidden**0.5 #kaiming initialization
        self.b_g = torch.zeros(n_hidden)
        
        self.W_io = torch.randn(n_input, n_hidden) / n_input**0.5 #kaiming initialization
        self.W_ho = torch.randn(n_hidden, n_hidden) / n_hidden**0.5 #kaiming initialization
        self.b_o = torch.zeros(n_hidden)

        
    def __call__(self, x):
        #(N, L, H_in) -> (N, L, H_out)
        #h_t -> (N, H_out)
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.n_hidden)
        c_t = torch.zeros(batch_size, self.n_hidden)
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
        return [self.W_ii, self.W_hi, self.b_i, self.W_if, self.W_hf, self.b_f, self.W_ig, self.W_hg, self.b_g, self.W_io, self.W_ho, self.b_o]

#GRU layer
class GRU:
    def __init__(self, n_input, n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden
        
        self.W_ir = torch.randn(n_input, n_hidden) / n_input**0.5 #kaiming initialization
        self.W_hr = torch.randn(n_hidden, n_hidden) / n_hidden**0.5 #kaiming initialization
        self.b_r = torch.zeros(n_hidden)
        
        self.W_iz = torch.randn(n_input, n_hidden) / n_input**0.5 #kaiming initialization
        self.W_hz = torch.randn(n_hidden, n_hidden) / n_hidden**0.5 #kaiming initialization
        self.b_z = torch.zeros(n_hidden)
        
        self.W_in = torch.randn(n_input, n_hidden) / n_input**0.5 #kaiming initialization
        self.W_hn = torch.randn(n_hidden, n_hidden) / n_hidden**0.5 #kaiming initialization
        self.b_in = torch.zeros(n_hidden)
        self.b_hn = torch.zeros(n_hidden)
    
    def parameters(self):
        return [self.W_ir, self.W_hr, self.b_r, self.W_iz, self.W_hz, self.b_z, self.W_in, self.W_hn, self.b_in, self.b_hn]
    
    def __call__(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.n_hidden)
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
    def __init__(self, num_features, eps=1e-5, momentum=0.1, training=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = training
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)
        
    def __call__(self, x):
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
        return [self.gamma, self.beta]

#Layer normalization
class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.num_features = normalized_shape
        self.eps = eps
        self.gamma = torch.ones(normalized_shape)
        self.beta = torch.zeros(normalized_shape)
        
    def __call__(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True, unbiased=False)
        self.out = (x - x_mean) / torch.sqrt(x_var + self.eps) * self.gamma + self.beta
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
