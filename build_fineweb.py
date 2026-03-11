"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

from datasets import load_dataset
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from tqdm import tqdm

#----------------------------------

DATASETS_DIR = 'datasets'
local_dir = 'edu_fineweb10B'
remote_name = 'sample-10BT'
SHARD_SIZE = int(1e8) # 100M tokens per shard, total of 100 shards

# Create the cache the local directory if it doesn't exist yet
data_cache_dir = os.path.join(os.path.dirname(__file__), DATASETS_DIR, local_dir)
os.makedirs(data_cache_dir, exist_ok=True)

# Download the dataset
# fw = load_dataset('HuggingFaceFW/fineweb-edu', name=remote_name, split='train')
# fw.to_csv("fineweb_10B.csv", index=False)
fw = load_dataset('csv', data_files="fineweb_10B.csv", split='train')

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # Tokenizes a single document and returns a numpy array of uint16 otkens
    tokens = [eot]  # The special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (tokens_np >= 0).all() and (tokens_np < 2**16).all(), "The token encoding is HUGE! Therefore, don't run."
    tokens_np_uint16 = tokens_np.astype(np.uint16)  # Want to keep token dictionary at 16 bit, as we're saving to shards
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)  # Pretty straightforward


# Tokenize all documents and write output shards, each of SHARD_SIZE tokens (last shard has remainder)
NPROCS = max(1, os.cpu_count() // 2)

with mp.Pool(NPROCS) as pool:
    shard_index = 0
    # Preallocate buffer to hold current shard
    all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < SHARD_SIZE:

            # Simply append tokens to the current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)

            # Update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=SHARD_SIZE, unit='tokens', desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))

        else:
            # I've never seen the first chunk of data set to be val, but that makes a lot of sense to me.
            split = 'val' if shard_index == 0 else 'train'  # I'm going to have to start doing this in future projects

            # Write the current shard and start a new one
            filename = os.path.join(data_cache_dir, f"edufineweb_{split}_{shard_index:6d}")

            # Split the document into whatever fits in this shard; the remainder goes to the next one.
            remainder = SHARD_SIZE - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None

            # populate the next shard with the leftovers of the current set of tokens
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]  # This is technically fine and more efficient, but man do I wish we were clearing this variable first...
            token_count = len(tokens) - remainder

    # Write any remaining tokens as the last shard
    if token_count != 0:
        split = 'val' if shard_index == 0 else 'train'  # I think this should never be 'val'
        filename = os.path.join(data_cache_dir, f"edufineweb_{split}_{shard_index:6d}")
        write_datafile(filename, all_tokens_np[:token_count])