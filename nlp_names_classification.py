from __future__ import unicode_literals, print_function, division 
from io import open 
import glob 
import os

def find_files(path): return glob.glob(path)
print(find_files('/content/data/names/*.txt'))

import unicodedata 
import string 

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('/content/data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
n_categories

#print(category_lines['Italian'][:5])

import torch 

# letter acccording to index
def letter_to_index(letter):
  return all_letters.find(letter)

# turn into [1, n_letters]
def letter_to_tensor(letter):
  tensor = torch.zeros(1, n_letters)
  tensor[0][letter_to_index(letter)] = 1

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors

def line_to_tensor(line):
  tensor = torch.zeros(len(line), 1, n_letters)
  for ln, letter in enumerate(line):
    tensor[ln][0][letter_to_index(letter)] = 1
  return tensor

#print(letter_to_tensor('J'))
#print(line_to_tensor('Jones').size())
# torch.Size([5, 1, 57])

import torch.nn as nn 

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


input = line_to_tensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)

def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(category_from_output(output))

import random 

def random_choice(l):
    return l[random.randint(0, len(l)-1)]

def random_training_example():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype = torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

'''
# 10 random samples 
for i in range(10):
  category, line, category_tensor, line_tensor = random_training_example()
  print('category =', category, '/ line =', line)
'''
criterion = nn.NLLLoss()
learning_rate = 0.005

def train(category_lines, line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
      output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    return output, loss.item()

import math 
import time 

n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0 
all_losses = []

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range (1, n_iters+1):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))

  # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker 

plt.figure()
plt.plot(all_losses)