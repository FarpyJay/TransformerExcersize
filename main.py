import numpy as np
import torch
from torch import nn

# This is an excerize to create a transformer neural network model from scratch.
# The reference video can be found at https://www.youtube.com/watch?v=kCc8FmEb1nY
class InputEmbeddings(nn.model):

    def __init__(self, model_dimension: int, vocab_size: int):
        super().__init__()
        self.model_dimension = model_dimension
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_dimension)

    def __call__(self, x):
        return self.embedding(x) * np.sqrt(self.model_dimension)


class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension: int, sequence_len: int, dropout: float):
        super().__init__()
        self.model_dimension = model_dimension
        self.sequence_len = sequence_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (sequence_len, model_dimension)
        encoding =torch.zeros(sequence_len, model_dimension)
        # Create a vector of shape (sequence_len, 1)
        position = torch.arange(0, sequence_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, model_dimension, 2).float() * (-np.log(10000.0) / model_dimension))