from typing import Tuple

import numpy as np
import pickle
import re
import torch
from tqdm import tqdm
from train import DerivativesModel

MAX_SEQUENCE_LENGTH = 30
TRAIN_URL = "https://drive.google.com/file/d/1ND_nNV5Lh2_rf3xcHbvwkKLbY2DmOH09/view?usp=sharing"


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    return int(true_derivative == predicted_derivative)


# --------- PLEASE FILL THIS IN --------- #
def predict(functions: str):
    # Extract the function and the variable.
    matches = re.match('d\((.*)\)/d(.*)', functions)
    eqn = matches.group(1)
    var = matches.group(2)

    # Tokenize the function.
    var_map = {var: 'var0'}
    var_index = 1
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
    var_map = {v: k for k, v in var_map.items()}

    # Load the vocabulary and index maps of the model.
    with open('input_token_to_index.pkl', 'rb') as f:
        input_token_to_index = pickle.load(f)
    with open('output_index_to_token.pkl', 'rb') as f:
        output_index_to_token = pickle.load(f)
    pad_token = '<pad>'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the input for the model.
    x = torch.tensor([input_token_to_index.get(token, pad_token) for token in tokens] + (MAX_SEQUENCE_LENGTH - len(tokens)) * [input_token_to_index[pad_token]]).to(device)
    # Load the model.
    model = torch.load('Derivatives.model', map_location=device)
    model = model.eval()
    with torch.no_grad():
        # Model expects input in batches. Create a batch of size 1 and pass it to the model.
        output = model(x.unsqueeze(0))
        output = output.view(-1, MAX_SEQUENCE_LENGTH, len(output_index_to_token)).squeeze()
        # Get the predicted indices.
        _, topi = output.topk(1, dim=1)
        topi = topi.squeeze()
        predicted_derivative = []
        # Map the indices to the tokens to form the predicted derivative.
        for index in topi:
            token = output_index_to_token[index.item()]
            if token == pad_token: break
            if token in var_map: token = var_map[token]
            predicted_derivative.append(token)
        return ''.join(predicted_derivative)

# ----------------- END ----------------- #


def main(filepath: str = "test.txt"):
    """load, inference, and evaluate"""
    # print(predict('d(126exp^(4w)*p^3)/dp'))
    functions, true_derivatives = load_file(filepath)
    predicted_derivatives = [predict(f) for f in tqdm(functions)]
    scores = [score(td, pd) for td, pd in zip(true_derivatives, predicted_derivatives)]
    print(np.mean(scores))


if __name__ == "__main__":
    main()
