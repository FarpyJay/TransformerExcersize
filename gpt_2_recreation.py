from dataclasses import dataclass
import inspect
import math
from torch import nn
import tiktoken

import torch.distributed as dist

from homemade_layers import *


def get_lr(iteration):
    max_lr = 1e-3
    min_lr = max_lr*0.1
    warmup_steps = 10
    max_steps = 50
    # 1: Linear warmup for warmput_iters steps
    if iteration < warmup_steps:
        return max_lr * (iteration+1) / warmup_steps
    # 2: If iteration > lr_decay_iters, return min learning rate
    if iteration > max_steps:
        return min_lr
    # 3: In between, use cosine decay down to min learning rate
    decay_ratio = (iteration - warmup_steps) / (max_steps-warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_emb % config.num_heads == 0
        # key, query, value projections for all heads, but effectively concatenated
        self.c_attn = nn.Linear(config.n_emb, 3*config.n_emb)
        # output projection
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.num_heads = config.num_heads
        self.n_emb = config.n_emb
        # not really a 'bias'. more of a mask, but following the OpenAI/HuggingFace naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    # This implementation of the multihead is kind of crazy. It's computationally optimized, but it's very hard to
    # follow the reshaping gymnastics that are going on.
    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_emb)
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_emb, dim=2)
        k = k.view(B, T, self.num_heads, C //self.num_heads).transpose(1,2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C //self.num_heads).transpose(1,2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C //self.num_heads).transpose(1,2)  # (B, nh, T, hs)


        # # attention (materializes the large (T,T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        # Use flash attention instead: https://arxiv.org/abs/2205.14135 and https://arxiv.org/abs/2307.08691
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1,2).contiguous().view(B, T, C) # reassemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_emb, 4*config.n_emb)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_emb, config.n_emb)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """Transformers: More than meets the eye. Transformers: Robots in disguise"""
    def __init__(self, config):
        super().__init__()
        assert config.num_heads > 0, "Number of heads must be a positive integer"
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.attn = CausalSelfAttention(config)
        # self.sas = MultiHead(config.num_heads, config.block_size, config.n_emb)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)
        # self.feed_forward = FeedForward(config.n_emb)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        # x = x + self.sas(self.lnorm1(x))
        x = x + self.mlp(self.ln_2(x))
        # x = x + self.feed_forward(self.lnorm2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    num_heads: int = 12 # number of heads
    n_emb: int = 768 # embedding dimension


class GPT(nn.Module):
    def __init__(self, config, device, master_process=True, ddp=False):
        super().__init__()

        # BFloat16 operations first became available starting with NVIDIA GTX 3000 series
        # If our GPU is compatible, then we use BFloat16 for computational efficiency
        gpu_model = torch.cuda.get_device_name(0)
        self.USE_BF16 = True if 'NVIDIA' in gpu_model and int(gpu_model.split()[-1]) >= 3000 else False

        self.config = config
        self.device = device
        self.master_process = master_process
        self.ddp = ddp

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emb),
            wpe = nn.Embedding(config.block_size, config.n_emb),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_emb)
        ))
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.optimizer = self.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

        self.apply(self._init_weights)

    # What a goofy little thing we have here. Why are we using norm with std=0.02?? Why not just use xavier or he???
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate=3e-4, device='cpu'):
        # Start with all of the candidate parameers (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optim groups. Any parameters that are 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"Using fused AdamW: {use_fused}")

        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def fit(self, data_loader, n_steps=50, accumulator_steps=1):
        for step in range(n_steps):  # increase number of steps for good results...
            self.optimizer.zero_grad()
            loss_accumulator = 0.0
            for acc_step in range(accumulator_steps):
                # sample a batch of data
                xb, yb = data_loader.next_batch()

                if self.ddp:  # If running a distributed parallel run, then use the synchronization flag
                    # WARNING: This is sketch, but it's less ugly than the official way PyTorch wants us to synchronize!
                    self.require_backward_grad_sync = acc_step == accumulator_steps - 1

                # evaluate the loss
                if self.USE_BF16:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits, loss = self.forward(xb, targets=yb)
                else:
                    logits, loss = self.forward(xb, targets=yb)
                loss /= accumulator_steps
                loss_accumulator += loss.detach()

                loss.backward()

            if self.ddp:
                dist.all_reduce(loss_accumulator, op=dist.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            # Determine and set the learning rate for this iteration
            lr = get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.optimizer.step()

            if self.master_process:
                print(f"Current step: {step}; Current loss: {loss_accumulator.item()}")


    @classmethod
    def from_pretrained(self, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" % model_type)

        # n_layer, num_heads, and n_emb are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, num_heads=12, n_emb=768),          # 124M params
            'gpt2-medium': dict(n_layer=24, num_heads=16, n_emb=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, num_heads=20, n_emb=1280),   # 774M params
            'gpt2-xl': dict(n_layer=48, num_heads=25, n_emb=1600),      # 1.5B params
        }[model_type]

        config_args['vocab_size'] = 50257  # always this size for GPT2 model checkpoints
        config_args['block_size'] = 1024   # always this size for GPT2 model checkpoints
        # create a from-scratch initialized miniGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        state_dict = model.state_dict()
        sd_keys = state_dict.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask/buffer, as it's not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # "basically, the openai checkpoints use a "Conv1D module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them" - Andrej Karpathy
        # "Ok buddy, I'm going to pretend I understand why transposed will successfully do that." - Me
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == state_dict[k].shape
                with torch.no_grad():
                    # Oh, I think I get it now. These particular modules are implemented with Conv1D?
                    # What's still a little nebulous for me is I forget why Conv1D only needs to be transposed,
                    # and why that's all that's necessary to conovert Conv1D to Linear
                    # Update: ok, here's Google AI's explanation: https://share.google/aimode/bRumgX3s9ryba3nH0
                    # Basically, OpenAI used Conv1D with kernal size of 1? Which is identical to Linear. But for
                    # whatever reason, this was a better implementation in Tensorflow because hardware reasons i guess
                    state_dict[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == state_dict[k].shape
                with torch.no_grad():
                    state_dict[k].copy_(sd_hf[k])
        return model


class DataLoaderLite:
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
