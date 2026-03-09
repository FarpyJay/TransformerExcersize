
# Base required libraries
import os
import time
import torch
from torch.nn import functional as F

# Libraries required for distributed runs
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# Local dependencies
from gpt_2_recreation import GPT, GPTConfig, DataLoaderLite

if __name__ == "__main__":

    ddp = int(os.environ.get('RANK', -1)) != -1 # Is this a distributed data parallel run?
    if ddp:
        # Use of DDP atm demands CUDA; we set the device appropriately according to rank
        assert torch.cuda.is_available(), "CUDA is necessary for DDP (probably)"
        init_process_group(backend='null')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # This process will do logging, checkpointing, etc.
    else:
        # Local, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # Attempt to autodetect device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
    use_huggingface = True
    do_training = True
    num_return_sequences = 5
    max_length = 30
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig(vocab_size=50304), device=device, master_process=master_process, ddp=ddp)

    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    if do_training:
        total_batch_size = 524288
        B, T = 4, 1024
        assert total_batch_size % (B*T) == 0, "total_batch_size must be divisible by B*T"
        grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
        if master_process:
            print(f"Total desired batch size: {total_batch_size}")
            print(f"--> Calculated gradient accumulation steps: {grad_accum_steps}")
        data_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, device=device)

        # Note: This command sets operations to use Tensor Float 32 precision. However, these operations did not become
        #       available on NVIDIA GPUs until the release of the GTX 3000 series. Therefore, the below command will
        #       have no effect if running on a CPU, an earlier NVIDIA GPU model, or an AMD RX Series GPU.
        # torch.set_float32_matmul_precision('high')

        t0 = time.time()  # Get ballpark time estimate.
        model.fit(data_loader, accumulator_steps=grad_accum_steps)
        torch.cuda.synchronize()
        t1 = time.time()

    if ddp:
        destroy_process_group()

    import sys; sys.exit(0)
    model.eval()
    model.to(device)


    tokens = enc.encode("hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
    x = tokens.to(device)
    # model = huggingface_model()

    # generate! right now, x is (B,T) where B=5, T=8
    # set the seed to 42
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits, _ = model(x)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1) # (8,1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (8,1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for ii in range(num_return_sequences):
        tokens = x[ii, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

    # if use_huggingface:
    #     responses = huggingface_response()
    # else:
    #     pass
    #
    # for response in responses:
    #     print(response['generated_text'])