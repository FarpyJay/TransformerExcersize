
from torch import nn
from torch.nn import functional as F

from helper_funcs import *
from homemade_layers import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device to use: %s' % device)

class Head(nn.Module):
    """ The meat and potatoes of attention. This thing is the opposite of ADHD"""
    def __init__(self, block_size, head_size, n_emb, decode_attn=True, dropout=0.2):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        if decode_attn:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.decode_attn = decode
        self.dropout = nn.Dropout(dropout)


    # I have commented on each portion of this function with my best interpretation of how to think of each of these
    # parts. Maybe I'll find that this is completely wrong, but I hope not. The last thing I want to do is piss off a
    # bunch of people who're smarter than me...
    def forward(self, x):
        B,T,C = x.shape
        # Key: What words/ideas might be most important in the prompt?
        k = self.key(x)  # (B,T,16)
        # Query: What words/ideas might be most important for my answer?
        q = self.query(x)  # (B,T,16)
        # Value: What words/ideas are usually most important in general?
        v = self.value(x) # (B,T,16); weighted aggregation of values
        # Weights: Something like the correlation between the interpretation of the prompt
        #          and the interpretation of possible answer words?
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5  # (B, T, T)
        if self.decode_attn:
            # Not allowed to see the future if the goal is specifically to try to predict the next word/idea
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)  # Probability distribution of what might be the best answer words/ideas
        wei = self.dropout(wei)
        out = wei @ v  # I wonder if a biased self.value is where the distinct vocabulary/tone of AI comes from?

        return out


class MultiHead(nn.Module):
    """Imagine the opposite of a kid with ADHD. Now imagine the opposite of a whole classroom of them."""
    def __init__(self, num_heads, block_size, n_emb, dropout=0.2):
        super().__init__()
        head_size = n_emb // num_heads
        self.heads = nn.ModuleList([Head(block_size, head_size, n_emb) for _ in range(num_heads)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)  # Note: See comment in def forward

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)

        # Andrej Karpathy includes a dropout layer here, but I'm not convinced that this is stable, since the output of
        # MultiHead is part of an input to a layer normalization. The problem is that when we calculate the xmean/xvar
        # for a given normalization, our gradient computation for gamma and beta may be incorrect due to not including
        # all of the weights to calculate xmean/xvar for the normalization. This is definitely an issue if we were to
        # use BatchNorm and were calculating a running xmean/xvar. I can't find any definitive literature on this for
        # LayerNorm. It may be that when you really work through the math, then this turns out to not be an issue at
        # all. But I'm just a lowly hobbyist right now and am not getting paid to do this, so I feel no obligation at
        # this time to go to all that trouble. I'd rather just go with what he found to work and maybe revisit later.
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    """This feels weird to implement in this way, but ok..."""
    def __init__(self, n_emb, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformers: More than meets the eye. Transformers: Robots in disguise"""
    def __init__(self, config):
        super().__init__()
        assert config.num_heads > 0, "Number of heads must be a positive integer"
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.sas = MultiHead(config.num_heads, config.block_size, config.n_emb)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.feed_forward = FeedForward(config.n_emb)


    def forward(self, x):
        x = x + self.sas(self.ln_1(x))
        x = x + self.feed_forward(self.ln_2(x))
        return x


class DecodeTransformer(nn.Module):
    """I know that these triple quotes at the beginning of classes can be used to automate documentation pages,
    but I have absolutely no idea how to actually do that, and this repo is nowhere near important enough for me or
    anyone else to care. Maybe I'll actually learn how to do that after I've finished my deep dive into the last five
    years of AI :)

    Also, I know that Andrej made vocab_size and n_emb global vars, but I want the user (me :P) to be able to pass
    whatever might make sense in the future. We ain't in Jupyter anymore. (And I've been traumatized by how global
    variables interact when you're trying to convert TF/PT code into a format that can be passed to some shitty custom
    API written by hardware engineers who've made a cool cutting edge accelerator but have never heard the term, UX"""
    def __init__(self, vocab_size, batch_size=64, block_size=256, n_emb=384, num_heads=6, n_layers=6):
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size
        self.token_emb_table = nn.Embedding(vocab_size, n_emb)
        self.position_emb_table = nn.Embedding(block_size, n_emb)
        # self.sa_heads = MultiHead(num_heads, block_size, n_emb//4, n_emb)  # self attention head
        # self.feed_forward = FeedForward(n_emb)
        self.blocks = nn.Sequential(*[Block(num_heads, block_size, n_emb) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)  # language modeling head
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)

        total_params = vocab_size * n_emb + block_size * n_emb
        total_params += sum(p.numel() for p in self.blocks.parameters())
        total_params += sum(p.numel() for p in self.ln_final.parameters())
        total_params += sum(p.numel() for p in self.lm_head.parameters())

        print("Total number of parameters in model: %s" % total_params)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_emb_table(idx)                                  # (B,T,C)
        position_emb = self.position_emb_table(torch.arange(T, device=device)) # (T,C)
        x = token_emb + position_emb                                           # (B,T,C)

        # x = self.sa_heads(x)                                                   # (B,T,C)
        # x = self.feed_forward(x)                                               # (B,T,C)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)                                               # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # Get the prediction
            logits, loss = self.forward(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # Becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def get_batch(self, data):
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    def fit(self, data, n_steps=5000):
        for steps in range(n_steps):  # increase number of steps for good results...
            # sample a batch of data
            xb, yb = self.get_batch(data)
            # evaluate the loss
            logits, loss = self.forward(xb, targets=yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if steps % 500 == 0:
                print("Current step: %s; Current loss: %s" % (steps, loss.item()))


if __name__=='__main__':
    preload = False
    train_model = True
    default_path = "shakespeare_gpt.pt"
    shakespeare_raw = get_words_from_file('input.txt')

    chars = sorted(list(set(shakespeare_raw)))
    char2index, index2char = get_letters_tables(chars)
    encode = lambda s: [char2index[c] for c in s]
    decode = lambda l: [index2char[i] for i in l]

    all_data = torch.tensor(encode(shakespeare_raw), dtype=torch.long)

    n = int(0.9*len(all_data))
    train_data = all_data[:n]
    val_data = all_data[n:]

    gpt_model = DecodeTransformer(len(chars))

    if preload:
        gpt_model.load_state_dict(torch.load(default_path, weights_only=True))

    gptm = gpt_model.to(device)

    if train_model:
        gptm.fit(train_data)
        torch.save(gptm.state_dict(), default_path)

    gptm.eval()

    # "Say the thing, B̶A̶R̶T̶ GPT"
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("".join(decode(gptm.generate(context, max_new_tokens=500)[0].tolist())))
