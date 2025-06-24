import torch
import matplotlib.pyplot as plt # for making figures
import torch.nn.functional as F
import time


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
        return [layer.parameters() for layer in self.layers]

    def append(self, layer):
        self.layers.append(layer)

#Embedding layer
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.randn(num_embeddings, embedding_dim)/num_embeddings**0.5

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
        self.out = x @ self.weight + self.bias
        return self.out
    def parameters(self):
        return [self.weight, self.bias]

#Tanh activation function
class Tanh:
    def __init__(self):
        pass
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

#softmax activation function
class Softmax:
    def __init__(self, dim = 1):
        self.dim = dim
    def __call__(self, x):
        self.out = torch.softmax(x, dim=self.dim)
        return self.out

#ReLU activation function
class ReLU:
    def __init__(self):
        pass
    def __call__(self, x):
        self.out = torch.relu(x)
        return self.out
