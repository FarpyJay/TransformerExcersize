import numpy as np
import os
import tiktoken
import torch

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# For shakespeare.txt or any other single-file txt training source
class DataLoaderText:
    def __init__(self, B, T, process_rank, num_processes, device, datafile='input.txt'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # At init, load tokens from disk and store them in memory
        with open(datafile, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens).to(device)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        # Advance the position in the tensor
        self.current_position += B*T*self.num_processes
        # If loading the next batch would be out of bounds, then reset
        if self.current_position + (B*T*self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y

# For much larger, already sharded/tokenized datasets
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split='train', master_process=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ['train', 'val']
        
        # Get the shard filenaes
        data_root = os.path.join(os.getcwd(), "datasets", "edu_fineweb10B")
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found{len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        buffer = self.tokens[self.current_position:self.current_position+self.B*self.T+1]
        x = buffer[:-1].view(self.B, self.T)
        y = buffer[1:].view(self.B, self.T)

        # Advance the position in the tensor
        self.current_position += self.B * self.T * self.num_processes
        # If loading the next batch would be out of bounds, then advance to the next shard
        if self.current_position + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard +1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y
