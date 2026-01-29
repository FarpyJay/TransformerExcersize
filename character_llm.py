import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

def get_words_from_file(filename='names.txt'):
    with open(filename, 'r') as namesfile:
        words = namesfile.read().splitlines()
    return words

def get_letters_tables(words):
    all_chars = sorted(set(''.join(words)))
    all_chars.insert(0, '.')
    letters_table = {s:i for i, s in enumerate(all_chars)}
    reverse_letters_table = {i:s for i, s in enumerate(all_chars)}
    return letters_table, reverse_letters_table

# A name generation model whose next character generation is done completely from the counts of char->next char pairs.
class SimpleCharLLM:
    def __init__(self, words, smoothing_factor=1):
        self.char2index, self.index2char = get_letters_tables(words)
        num_chars= len(self.char2index)
        self.num_chars = num_chars
        self.bigram_tensor = self._get_brigram_tensor() + smoothing_factor # Smoothing factor prevents inf
        self.prob_vector = self.bigram_tensor.float() / self.bigram_tensor.sum(dim=1, keepdim=True)

    def _get_brigram_tensor(self):
        bigram_frequency = torch.zeros((self.num_chars, self.num_chars), dtype=torch.int32)
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                index1 = self.char2index[ch1]
                index2 = self.char2index[ch2]
                bigram_frequency[index1, index2] += 1

        return bigram_frequency

    def __call__(self, number_of_names, seed=123):
        name_list = list()
        gen = torch.Generator().manual_seed(seed)
        for ii in range(number_of_names):
            current_index = 0
            name = ''
            while True:
                generated_index = (
                    torch.multinomial(self.prob_vector[current_index], num_samples=1, replacement=True,
                                      generator=gen).item())
                if generated_index == 0:
                    break
                name += self.index2char[generated_index]
                current_index = generated_index

            name_list.append(name)
        return name_list

# A name generation model whose next character generation is done from a trained shallow neural network
class NNCharLLM:
    def __init__(self, words, seed=123):
        self.char2index, self.index2char = get_letters_tables(words)
        self.xs, self.ys = self._create_bigram_dataset(words)
        self.gen = torch.Generator().manual_seed(seed)
        self.W = torch.randn((len(self.char2index),len(self.char2index)), generator=self.gen, requires_grad=True)

    def _create_bigram_dataset(self, words):
        xs, ys = [], []
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                index1 = self.char2index[ch1]
                index2 = self.char2index[ch2]
                xs.append(index1)
                ys.append(index2)
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)
        return xs, ys

    def fit(self, num_epochs=100, lr=50, regularization=0.005):

        for epoch in range(num_epochs):
            # Forward pass
            logits = F.one_hot(self.xs, num_classes=len(self.char2index)).float() @ self.W
            counts = logits.exp()
            prob_vector = counts / counts.sum(1, keepdim=True)
            loss = -prob_vector[torch.arange(len(prob_vector)), self.ys].log().mean() + regularization*(self.W**2).mean()

            # Backward pass
            self.W.grad = None # Set gradient to zero
            loss.backward()
            self.W.data -= lr * self.W.grad
            print("Current loss: %s" % loss.item())

        return

    def __call__(self, number_of_names):
        name_list = list()
        for ii in range(number_of_names):
            current_index = 0
            name = ''
            while True:
                # Note: if i make a video on this, i neglected the square brackets around current index and stared
                #       at my screen for like 10 minutes before I figured out waht i did wrong
                logits = F.one_hot(torch.tensor([current_index]), num_classes=len(self.char2index)).float() @ self.W
                counts = logits.exp()
                prob_vector = counts / counts.sum(1, keepdim=True)
                generated_index = \
                    torch.multinomial(prob_vector, num_samples=1, replacement=True, generator=self.gen).item()
                if generated_index == 0:
                    break
                name += self.index2char[generated_index]
                current_index = generated_index
            name_list.append(name)
        return name_list


def get_log_likelihood(words, char2index, prob_vector):
    log_likelihood = 0
    n = 0
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            index1 = char2index[ch1]
            index2 = char2index[ch2]
            prob = prob_vector[index1, index2]
            log_likelihood += torch.log(prob)
            n += 1
    return log_likelihood, n

if __name__ == '__main__':
    words = get_words_from_file()
    # naive_name_generator = SimpleCharLLM(words)
    # print(naive_name_generator(30))

    neural_network_name_generator = NNCharLLM(words)
    neural_network_name_generator.fit()
    print(neural_network_name_generator(30))
