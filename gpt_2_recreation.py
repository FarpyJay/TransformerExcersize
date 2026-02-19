
import torch
from torch import nn
from torch.nn import functional as F

from helper_funcs import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Transformer(nn.Module):
    # I know that Andrej made vocab_size and n_emb global vars, but I want the
    # user (me :P) to be able to pass whatever might make sense in the future.
    def __init__(self, block_size, vocab_size, n_emb):
        super().__init__()

        self.token_emb_table = nn.Embedding(vocab_size, n_emb)
        self.position_emb_table = nn.Embedding(block_size, n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_emb_table(idx)  # (B,T,C)
        position_emb = self.position_emb_table(torch.arange(T, device=device)) # (T,C)
        x = token_emb + position_emb
        logits = self.lm_head(x)       # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)



def get_batch(data, block_size=8, batch_size=4):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

if __name__=='__main__':
    # shakespeare_raw = get_words_from_file('input.txt')
    #
    # chars = sorted(list(set(shakespeare_raw)))
    # vocab_size = len(chars)
    # char2index, index2char = get_letters_tables(chars)
    # encode = lambda s: [char2index[c] for c in s]
    # decode = lambda l: [index2char[i] for i in l]
    #
    # all_data = torch.tensor(encode(shakespeare_raw), dtype=torch.long)
    # n = int(0.9*len(all_data))
    # train_data = all_data[:n]
    # val_data = all_data[n:]
    # x_train, y_train = get_batch(train_data)
    # x_val, y_val = get_batch(val_data)
    #
    B, T, C = 4, 8, 2  # batch, time (block_size), channels (don't understand channels yet
    x = torch.randn(B,T,C)

    tril = torch.tril(torch.ones(T, T))
    avg_muliplier = torch.zeros((T,T))
    avg_muliplier = avg_muliplier.masked_fill(tril == 0, float('-inf'))
    avg_multiplier = F.softmax(avg_muliplier, dim=-1)

    x_bag_of_words = avg_multiplier @ x
