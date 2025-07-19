# Makemoremore

A comprehensive neural network implementation from scratch for language modeling tasks. This project implements various neural network architectures including GPT (Generative Pre-trained Transformer), RNN, LSTM, GRU, and MLP models, all built from the ground up without relying on PyTorch's built-in layers.

## 🚀 Features

- **Complete GPT Implementation**: Full transformer architecture with self-attention, multi-head attention, and causal masking
- **Multiple RNN Variants**: RNN, LSTM, and GRU implementations from scratch
- **MLP Models**: Multi-layer perceptron for sequence modeling
- **Custom Neural Network Modules**: All layers implemented from scratch including:
  - Linear layers with Kaiming initialization
  - Embedding layers
  - Activation functions (ReLU, Tanh, Softmax)
  - Normalization layers (BatchNorm1d, LayerNorm)
  - Dropout for regularization
  - Sequential container for easy model composition
- **Optimization**: Custom Adam optimizer implementation
- **Tokenization**: Support for tiktoken tokenization (GPT-4 tokenizer)
- **Training Scripts**: Complete training loops for all model types
- **Jupyter Notebooks**: Interactive examples and demonstrations

## 📁 Project Structure

```
makemoremore/
├── code/                    # Core implementation files
│   ├── GPT.py              # Complete GPT model implementation
│   ├── module.py            # Neural network building blocks
│   ├── optim.py             # Custom optimizer implementations
│   ├── dataset.py           # Dataset utilities
│   ├── TrainGPT.py          # GPT training script
│   ├── TrainRNN.py          # RNN training script
│   ├── TrainLSTM.py         # LSTM training script
│   ├── TrainGRU.py          # GRU training script
│   └── TrainMLP.py          # MLP training script
├── data/                    # Data directory
├── test/                    # Test files
├── main.ipynb              # Main demonstration notebook
├── test.ipynb              # Testing notebook
├── tokenizer.ipynb         # Tokenization examples
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd makemoremore
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

- `torch>=1.9.0` - PyTorch for tensor operations
- `numpy>=1.21.0` - Numerical computing
- `tiktoken>=0.5.0` - Tokenization for language models

## 🎯 Quick Start

### Running the Main Demo

```bash
jupyter notebook main.ipynb
```

### Training a GPT Model

```bash
cd code
python TrainGPT.py
```

### Training Other Models

```bash
cd code
python TrainRNN.py    # For RNN
python TrainLSTM.py   # For LSTM
python TrainGRU.py    # For GRU
python TrainMLP.py    # For MLP
```

## 🧠 Model Architectures

### GPT (Generative Pre-trained Transformer)

The GPT implementation includes:
- **Self-Attention**: Scaled dot-product attention with causal masking
- **Multi-Head Attention**: Multiple attention heads with concatenation
- **Feed-Forward Networks**: Two-layer networks with ReLU activation
- **Residual Connections**: Skip connections for better gradient flow
- **Layer Normalization**: Normalization for training stability
- **Position Embeddings**: Learnable position encodings
- **Token Embeddings**: Learnable token representations

### RNN Variants

- **RNN**: Basic recurrent neural network
- **LSTM**: Long Short-Term Memory with forget, input, and output gates
- **GRU**: Gated Recurrent Unit with update and reset gates

### MLP

- Multi-layer perceptron for sequence modeling
- Configurable hidden layers and activation functions

## 🔧 Custom Modules

All neural network components are implemented from scratch:

- **Linear Layer**: Fully connected layer with Kaiming initialization
- **Embedding Layer**: Dense representations for discrete inputs
- **Activation Functions**: ReLU, Tanh, Softmax
- **Normalization**: BatchNorm1d, LayerNorm
- **Regularization**: Dropout
- **Sequential Container**: Easy layer composition

## 📊 Training

Each model type has its own training script with:
- Configurable hyperparameters
- Loss tracking and evaluation
- Text generation capabilities
- Device support (CPU/CUDA)

### Example Training Configuration

```python
# GPT Training Parameters
block_size = 512      # Context length
batch_size = 64       # Batch size
max_iters = 5000      # Training iterations
learning_rate = 3e-4  # Learning rate
n_embd = 384         # Embedding dimension
n_heads = 6          # Attention heads
n_layers = 6         # Transformer layers
```

## 📈 Usage Examples

### Training a Model

```python
from code.GPT import GPT
from code.optim import Adam

# Initialize model
model = GPT(vocab_size, n_embd, n_heads, n_layers, block_size)
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Text Generation

```python
# Generate text with trained model
context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=100)
```

## 🧪 Testing

Run the test notebook to verify implementations:

```bash
jupyter notebook test.ipynb
```

## 📝 Jupyter Notebooks

- **main.ipynb**: Main demonstration with character-level language modeling
- **test.ipynb**: Comprehensive testing of all implementations
- **tokenizer.ipynb**: Tokenization examples and experiments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Nazar Korniichuk**

## 🙏 Acknowledgments

- Inspired by Andrej Karpathy's "makemore" project
- Built for educational purposes in understanding neural network internals
- All implementations are from scratch for learning purposes

---

**Note**: This project is designed for educational purposes to understand neural network internals. All implementations are built from scratch without relying on PyTorch's built-in layers, making it an excellent resource for learning how neural networks work under the hood.
