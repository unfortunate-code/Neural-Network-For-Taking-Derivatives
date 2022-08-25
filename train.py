import itertools
import math
import pickle
import random
import re
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchsummaryX import summary
from tqdm import tqdm

# Extracts the input function, variable to be differentiated on and the derivative from the data file.
def simplify_equation(line):
  x, y = line.strip().split('=')
  matches = re.match('d\((.*)\)/d(.*)', x)
  return matches.group(1), matches.group(2), y

# Tokenizes the equations. Extracts variables, functions, numbers and returns the tokens.
def tokenize_equations(eqn, var, var_map):
  if not var_map:
    var_map[var] = 'var0'
    var_index = 1
  else:
    var_index = int(sorted(list(var_map.values()))[-1][-1]) + 1
  curr = ''
  tokens = []
  for i in range(len(eqn)):
    if 'a' <= eqn[i] <='z' or 'A' <= eqn[i] <= 'Z':
      curr += eqn[i]
    else:
      if curr:
        if len(curr) == 1:
          if curr in var_map:
            tokens.append(var_map[curr])
          else:
            var_map[curr] = 'var' + str(var_index)
            var_index += 1
            tokens.append(var_map[curr])
        else:
          tokens.append(curr)
        curr = ''
      tokens.append(eqn[i])
  return tokens, var_map

class DerivativesDataset(Dataset):
  def __init__(self, data, input_token_to_index, output_token_to_index, pad_token, max_sequence_length):
    self.data = data
    self.input_token_to_index = input_token_to_index
    self.output_token_to_index = output_token_to_index
    self.pad_token = pad_token
    self.max_sequence_length = max_sequence_length

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    x, y, map = self.data[index]
    x = torch.tensor([self.input_token_to_index[token] for token in x] + (self.max_sequence_length - len(x)) * [self.input_token_to_index[self.pad_token]])
    y = torch.tensor([self.output_token_to_index[token] for token in y] + (self.max_sequence_length - len(y)) * [self.output_token_to_index[self.pad_token]])
    return x, y

# The model. It is a FCNN which takes the tokenized function as input and returns a distribution of derivative tokens for the output.
class DerivativesModel(nn.Module):
  def __init__(self, input_size, sequence_length, output_size, hidden_size, dropout):
    super(DerivativesModel, self).__init__()
    self.embedding_layer = nn.Embedding(input_size, hidden_size)
    self.dropout1 = nn.Dropout(dropout)
    self.linear1 = nn.Linear(hidden_size * sequence_length, math.floor(hidden_size * sequence_length * 1.5))
    self.linear2 = nn.Linear(math.floor(hidden_size * sequence_length * 1.5), hidden_size * sequence_length * 2)
    self.dropout2 = nn.Dropout(dropout)
    self.linear3 = nn.Linear(hidden_size * sequence_length * 2, math.floor(hidden_size * sequence_length * 1.5))
    self.linear4 = nn.Linear(math.floor(hidden_size * sequence_length * 1.5), output_size * sequence_length)
    self.output_size = output_size
    torch.nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in')
    torch.nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in')
    torch.nn.init.kaiming_normal_(self.linear3.weight, mode='fan_in')

  def forward(self, x):
    embedding = self.embedding_layer(x)
    embedding = embedding.view(x.shape[0], -1)
    embedding = self.dropout1(embedding)
    output = self.linear1(embedding)
    output = torch.nn.functional.relu(output)
    output = self.linear2(output)
    output = torch.nn.functional.relu(output)
    output = self.dropout2(output)
    output = self.linear3(output)
    output = torch.nn.functional.relu(output)
    output = self.linear4(output)
    output = output.view(-1, self.output_size)
    return output

# Training loop for the model.
def train(dataloader, model, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
      print('Epoch', epoch)
      total_loss = []
      for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x)
        y = y.view(-1)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
      print(sum(total_loss) / len(total_loss))

# Evaluates the model and returns the accuracy.
def evaluate(dataloader, model, max_sequence_length, device):
  model = model.eval()
  count = correct_count = 0
  with torch.no_grad():
    for x, y in tqdm(dataloader):
      count += len(x)
      x, y = x.to(device), y.to(device)
      output = model(x)
      output = output.view(-1, max_sequence_length, len(output_vocabulary))
      _, topi = output.topk(1, dim = 2)
      topi = topi.squeeze(2)
      correct_count += torch.sum(torch.all(topi == y, 1)).item()
  return correct_count / count

# Gets the index mappings for the input and output vocabulary.
def get_index_token_maps(input_vocabulary, output_vocabulary):
    input_token_to_index = {}
    input_index_to_token = {}
    index = 0
    for token in input_vocabulary:
      input_token_to_index[token] = index
      input_index_to_token[index] = token
      index += 1
    output_token_to_index = {}
    output_index_to_token = {}
    index = 0
    for token in output_vocabulary:
      output_token_to_index[token] = index
      output_index_to_token[index] = token
      index += 1
    return input_token_to_index, input_index_to_token, output_token_to_index, output_index_to_token

# Gets the train, test and validation split indices from the data. 90% is train, 5% is validation and 5% is test.
def get_splits(data):
    all_indices = set(range(len(data)))
    train_indices = set(random.sample(all_indices, 9 * len(all_indices) // 10))
    all_indices -= train_indices
    val_indices = set(random.sample(all_indices, len(all_indices) // 2))
    all_indices -= val_indices
    test_indices = all_indices
    return train_indices, val_indices, test_indices

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data.
    print('Loading data...')
    with open('train.txt', 'r') as f:
        data = [simplify_equation(line) for line in f.readlines()]

    # Tokenize the data.
    print('Tokenizeing data...')
    tokenized_data = []
    for x, v, y in data:
      x_tokens, var_map = tokenize_equations(x, v, {})
      y_tokens, var_map = tokenize_equations(y, v, var_map)
      tokenized_data.append((x_tokens, y_tokens, {v: k for k, v in var_map.items()}))

    # Extract the input (function) and output (derivative) vocabulary and create a padding token.
    input_vocabulary = set(itertools.chain.from_iterable([x for x, y, _ in tokenized_data]))
    output_vocabulary = set(itertools.chain.from_iterable([y for x, y, _ in tokenized_data]))
    pad_token = '<pad>'
    input_vocabulary.add(pad_token)
    output_vocabulary.add(pad_token)
    input_token_to_index, input_index_to_token, output_token_to_index, output_index_to_token = get_index_token_maps(input_vocabulary, output_vocabulary)

    # Dump the index mappings for the input (function) and output (derivative) vocabulary
    with open('input_token_to_index.pkl', 'wb') as f:
        pickle.dump(input_token_to_index, f)
    with open('input_index_to_token.pkl', 'wb') as f:
        pickle.dump(input_index_to_token, f)
    with open('output_token_to_index.pkl', 'wb') as f:
        pickle.dump(output_token_to_index, f)
    with open('output_index_to_token.pkl', 'wb') as f:
        pickle.dump(output_index_to_token, f)

    print('Creating splits and setting up the model...')
    # Split the data into train, validation and test
    train_indices, val_indices, test_indices = get_splits(data)
    train_data = [tokenized_data[i] for i in train_indices]
    test_data = [tokenized_data[i] for i in test_indices]
    val_data = [tokenized_data[i] for i in val_indices]

    # Setting constants.
    learning_rate = 0.001
    epochs = 100
    hidden_size = 22
    dropout = 0.2
    max_sequence_length = 30
    batch_size = 1024

    # Creating the training dataloader
    train_dataset = DerivativesDataset(train_data, input_token_to_index, output_token_to_index, pad_token, max_sequence_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Setting up the model
    model = DerivativesModel(len(input_vocabulary), max_sequence_length, len(output_vocabulary), hidden_size, dropout)
    model = model.train().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print('Training...')
    # Train the model and save the model state.
    train(train_dataloader, model, criterion, optimizer, epochs, device)
    # Uncomment this to save the model.
    torch.save(model, 'Derivatives.model')

    #Evaluations.
    print('Evaluating Validation data')
    val_dataset = DerivativesDataset(train_data, input_token_to_index, output_token_to_index, pad_token)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    evaluate(val_dataloader, model, max_sequence_length, device)

    print('Evaluating Testing data')
    test_dataset = DerivativesDataset(train_data, input_token_to_index, output_token_to_index, pad_token)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    evaluate(test_dataloader, model, max_sequence_length, device)

    print('Evaluating training data')
    evaluate(train_dataloader, model, max_sequence_length, device)


if __name__ == '__main__':
    main()
