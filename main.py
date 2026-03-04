import time
import torch
from torch.nn import functional as F
from gpt_2_recreation import GPT, GPTConfig, DataLoaderLite
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    use_huggingface = True
    do_training = True
    num_return_sequences = 5
    max_length = 30
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())

    model.to(device)
    model = torch.compile(model)

    if do_training:
        B, T = 4, 1024
        data_loader = DataLoaderLite(B, T)

        # Note: This command sets operations to use Tensor Float 32 precision. However, these operations did not become
        #       available on NVIDIA GPUs until the release of the GTX 3000 series. Therefore, the below command will
        #       have no effect if running on a CPU, an earlier NVIDIA GPU model, or an AMD RX Series GPU.
        # torch.set_float32_matmul_precision('high')

        t0 = time.time()  # Get ballpark time estimate.
        model.fit(data_loader)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"we made it through a few steps of training. Yay! Training time: {(t1-t0) / 360} hours")

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