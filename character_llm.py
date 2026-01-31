import matplotlib.pyplot as plt
import random
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
    def __init__(self, words, block_size=3, split_data=True, seed=123):
        self.block_size = max(block_size, 1)
        self.char2index, self.index2char = get_letters_tables(words)

        random.seed(seed)
        random.shuffle(words)
        if split_data:
            split_point_1 = int(0.8*len(words))
            split_point_2 = int(0.9*len(words))
            self.x_train, self.y_train = self._create_bigram_dataset(words[:split_point_1])
            self.x_val, self.y_val = self._create_bigram_dataset(words[split_point_1:split_point_2])
            self.x_test, self.y_test = self._create_bigram_dataset(words[split_point_2:])
        else:
            self.x_train, self.y_train = self._create_bigram_dataset(words)
            self.x_val, self.y_val = self.x_train, self.y_train
            self.x_test, self.y_test = self.x_train, self.y_train

        self.gen = torch.Generator().manual_seed(seed)
        self.emb_vector = torch.randn((len(self.char2index),10), generator=self.gen, requires_grad=True)
        self.W1 = torch.randn((30, 200), generator=self.gen, requires_grad=True)
        self.b1 = torch.randn(200, generator=self.gen, requires_grad=True)
        self.W2 = torch.randn((200, len(self.char2index)), generator=self.gen, requires_grad=True)
        self.b2 = torch.randn(len(self.char2index), generator=self.gen, requires_grad=True)
        self.parameters = [self.emb_vector, self.W1, self.b1, self.W2, self.b2]

    def _create_bigram_dataset(self, words):
        xs, ys = [], []
        for w in words:
            context = [0] * self.block_size
            chs = list(w) + ['.']
            for ch in chs:
                index = self.char2index[ch]
                xs.append(context)
                ys.append(index)
                context = context[1:] + [index]
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)
        return xs, ys


    def fit(self, num_epochs=200000, lr=0.1, minibatch_size=32, regularization=0.005):

        # lre = torch.linspace(-3, 0, num_epochs)
        # lrs = 10**(lre)

        for epoch in range(num_epochs):
            # lr = lrs[epoch]
            if type(minibatch_size) is int and minibatch_size > 0:
                assert minibatch_size < self.x_train.shape[0],\
                    ("minibatch size of %i has been chosen, but sample size is only %i" % (minibatch_size, self.xs.shape[0]))
                xs_batch = self.x_train[torch.randint(0, self.x_train.shape[0], (minibatch_size,))]
                ys_batch = self.y_train[torch.randint(0, self.y_train.shape[0], (minibatch_size,))]
            else:
                xs_batch, ys_batch = self.x_train, self.y_train

            # Forward pass
            embedding = self.emb_vector[xs_batch]
            h = torch.tanh(embedding.view(-1, 30) @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            loss = F.cross_entropy(logits, ys_batch)

            # Backward pass
            for param in self.parameters:
                param.grad = None
            loss.backward()

            # Note: I missed this in the first time through. Maybe this solves the loss issue???
            lr_to_use = lr if epoch < 100000 else lr * 0.1
            for param in self.parameters:
                param.data -= lr_to_use * param.grad

        train_loss = F.cross_entropy(
            (torch.tanh(self.emb_vector[self.x_train].view(-1, 30) @ self.W1 + self.b1) @ self.W2 + self.b2), self.y_train)
        val_loss = F.cross_entropy(
            (torch.tanh(self.emb_vector[self.x_val].view(-1, 30) @ self.W1 + self.b1) @ self.W2 + self.b2), self.y_val)
        print("Current train loss: %s\nCurrent val loss: %s" % (train_loss.item(), val_loss.item()))

        return

    def plot_emb_vector(self):
        plt.figure(figsize=(8,8))
        plt.scatter(self.emb_vector[:, 0].data, self.emb_vector[:, 1].data, s=200)
        for ii in range(self.emb_vector.shape[0]):
            plt.text(
                self.emb_vector[ii, 0].item(),
                self.emb_vector[ii, 1].item(),
                self.index2char[ii],
                ha='center',
                va='center',
                color='white'
            )
        plt.grid('minor')
        plt.savefig('emb_scatter.png')

    def __call__(self, number_of_names):
        name_list = list()
        for ii in range(number_of_names):
            context = [0] * self.block_size
            name = ''
            while True:
                embedding = self.emb_vector[context]
                h = torch.tanh(embedding.view(-1, 30) @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                counts = logits.exp()
                prob_vector = counts / counts.sum(1, keepdim=True)
                generated_index = \
                    torch.multinomial(prob_vector, num_samples=1, replacement=True, generator=self.gen).item()
                if generated_index == 0:
                    break
                name += self.index2char[generated_index]
                context = context[1:] + [generated_index]
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
    # neural_network_name_generator.fit(num_epochs=1, lr=0, minibatch_size=None)
    print(neural_network_name_generator(30))
