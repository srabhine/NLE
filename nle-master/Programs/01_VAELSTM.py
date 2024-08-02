import os
os.environ['KERAS_BACKEND'] = 'torch'

import keras
from keras import layers

import numpy as np

#from modules.models    import VAE
#from modules.layers    import SamplingLayer
#from modules.callbacks import ImagesCallback
#from modules.datagen   import MNIST

#import matplotlib.pyplot as plt
import scipy.stats
import sys

import fidle
import pandas as pd
from torch.utils.data import DataLoader

import time
import torch
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence 
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loss_weights  = [1,.06]

scale         = .2
seed          = 123

batch_size    = 64
fit_verbosity = 1

#fidle.override('latent_dim', 'loss_weights', 'scale', 'seed', 'batch_size', 'epochs', 'fit_verbosity')

train_data = pd.read_csv('/dm4i/work/towards.explainable.nlp/data/pcmag/train.csv')
val_data =pd.read_csv('/dm4i/work/towards.explainable.nlp/data/pcmag/valid.csv')
test_data = pd.read_csv('/dm4i/work/towards.explainable.nlp/data/pcmag/test.csv')

#print("train dataframe shape", train_data.shape)
#print("test dataframe shape", test_data.shape)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



train_data['combined_text'] = train_data[['review_text', 'positive_comment', 'negative_comment', 'neural_comment']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
val_data['combined_text'] = val_data[['review_text', 'positive_comment', 'negative_comment', 'neural_comment']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
test_data['combined_text'] = test_data[['review_text', 'positive_comment', 'negative_comment', 'neural_comment']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Tokenize the texts
train_sequences = [tokenizer.encode(text,truncation=True ,add_special_tokens=True) for text in train_data['combined_text']]
val_sequences = [tokenizer.encode(text, truncation=True,add_special_tokens=True) for text in val_data['combined_text']]
test_sequences = [tokenizer.encode(text,truncation=True ,add_special_tokens=True) for text in test_data['combined_text']]

# Pad the sequences
train_data = pad_sequence([torch.tensor(seq) for seq in train_sequences], batch_first=True, padding_value=0)
val_data = pad_sequence([torch.tensor(seq) for seq in val_sequences], batch_first=True, padding_value=0)
test_data = pad_sequence([torch.tensor(seq) for seq in test_sequences], batch_first=True, padding_value=0)

# Get the vocabulary size
NB_WORDS = len(tokenizer.vocab) + 1  # +1 for zero padding
output_dim= NB_WORDS

#print('Found %s unique tokens' % len(tokenizer.vocab))
#print('Shape of train data tensor:', train_data.shape)
#print('Shape of test data tensor:', test_data.shape)

# Create a mapping from index to word
index2word = {v: k for k, v in tokenizer.vocab.items()}

# Prepare GloVe embeddings
GLOVE_EMBEDDING = '/dm4i/work/VAE-Text-Generation/glove.6B.300d.txt'
EMBEDDING_DIM = 300

# Load GloVe embeddings
glove_embeddings = {}
with open(GLOVE_EMBEDDING, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
        glove_embeddings[word] = coefs

# Create embedding matrix

glove_embedding_matrix = torch.zeros(NB_WORDS, EMBEDDING_DIM).to(device)
for word, idx in tokenizer.vocab.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        glove_embedding_matrix[idx] = embedding_vector.to(device)
    else:
        # Use embedding for 'unk' token if word not found in GloVe
        glove_embedding_matrix[idx] = glove_embeddings.get('unk', torch.zeros(EMBEDDING_DIM)).to(device)





# Count null word embeddings
null_word_embeddings = torch.sum(torch.sum(glove_embedding_matrix, dim=1) == 0)
#print(f'Found {len(glove_embeddings)} word vectors.')
#print(f'Null word embeddings: {null_word_embeddings.item()}')

# Create PyTorch datasets and data loaders
train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)
test_dataset = TensorDataset(test_data)

#print("Total number of samples in test dataset:", len(test_dataset))

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(type(val_loader))

#print("train loader shape", type(train_loader))
#print("test loader shape", type(test_loader))

### VAE Model ###

class VAEEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, glove_embeddings, intermediate_dim, latent_dim, num_layers=10):
        super(VAEEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(glove_embeddings, freeze=True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=intermediate_dim, 
                            num_layers=num_layers,bidirectional=True, batch_first=True) #if num_layers>1 add dropout=0.2
        self.fc_mu = nn.Linear(2 * intermediate_dim, latent_dim)
        self.fc_logvar = nn.Linear(2 * intermediate_dim, latent_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        # Assuming LSTM is bidirectional, we take the last hidden state from both directions
        x = x[:, -1, :]
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, intermediate_dim, sequence_length, vocab_size, num_layers=10):
        super(VAEDecoder, self).__init__()
        self.sequence_length = sequence_length
        #self.repeated_context = nn.RepeatVector(sequence_length)
        self.intermediate_dim= intermediate_dim
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=intermediate_dim,
                            num_layers=num_layers, batch_first=True) #if num_layers>1 add dropout=0.2
        self.fc = nn.Linear(intermediate_dim, vocab_size)

    def forward(self, z):
       # z = self.repeated_context(z)
        z= z.unsqueeze(1).repeat(1,self.sequence_length, 1)
        z, _ = self.lstm(z)
        z = self.fc(z)
        return z

class VAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, glove_embeddings, sequence_length, intermediate_dim, latent_dim, num_layers=3):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(vocab_size, embedding_dim, glove_embeddings, intermediate_dim, latent_dim, num_layers)
        self.decoder = VAEDecoder(latent_dim, intermediate_dim, sequence_length, vocab_size, num_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, kl_weight=0.01):
    recon_loss = nn.CrossEntropyLoss()(recon_x.transpose(1, 2), x)  # recon_x needs to be [batch, classes, seq_len]
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss


vocab_size = len(tokenizer.vocab)  # Define vocabulary size
embedding_dim = 300  # Define embedding dimension
#sequence_length = 100 
sequence_length=512
intermediate_dim = 256
latent_dim = 64
glove_embeddings = torch.randn(vocab_size, embedding_dim)  # Placeholder for GloVe embeddings

model = VAE(vocab_size, embedding_dim, glove_embeddings, sequence_length, intermediate_dim, latent_dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


## catching error ##

try:
    max_index = max(batch[0].max().item() for batch in train_loader)  # Adjust depending on the structure of your batch
    print(f"Maximum token index in training data: {max_index}")
    print(f"Embedding matrix size: {glove_embedding_matrix.size(0)}")
except Exception as e:
    print("Error checking max index:", e)

# Check if the max index is within the bounds of the embedding matrix
if max_index >= glove_embedding_matrix.size(0):
    print("Warning: Max index exceeds the size of the embedding matrix. There might be an out-of-bounds error during training.")

for batch in val_loader:
    print(type(batch), batch)


# Training loop
epochs = 4
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, data in progress_bar:
        data = data[0].to(device)  
        if data.max() >= glove_embedding_matrix.size(0):
            raise ValueError(f"Found token index {data.max()} exceeding embedding matrix size {glove_embedding_matrix.size(0)}")
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % fit_verbosity == 0:
            print('Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, total_loss / len(train_loader.dataset)))

print("Training completed successfully.")

def tokens_to_text(token_ids, index2word):
    """ Convert a list of token IDs to their corresponding words, filtering out special tokens. """
    return ' '.join([index2word.get(token.item(), '<UNK>') for token in token_ids if index2word.get(token.item(), '<UNK>') not in ['<UNK>', 'pad', '[CLS]', '[SEP]']])

model.eval()

sent_idx = 2  
with torch.no_grad():
    for i, input_data in enumerate(val_loader):  # Directly get input_data without unpacking
        if i == 0:  # Just taking the first batch for simplicity
            
            #input_data = input_data.to(device)
            input_data = batch[0].to(device)

            # Forward pass through the model
            logits, _, _ = model(input_data)
            probabilities = torch.softmax(logits, dim=-1)
            reconstructed_indexes = torch.argmax(probabilities, dim=-1)
            
            # Handle dimensions if necessary
            if reconstructed_indexes.dim() > 2:
                reconstructed_indexes = reconstructed_indexes.squeeze()

            # Convert the indices of the first sentence to text
            reconstructed_text = tokens_to_text(reconstructed_indexes[sent_idx], index2word)
            original_text = tokens_to_text(input_data[sent_idx], index2word)

            print('Reconstructed:', reconstructed_text)
            print('Original:', original_text)
            break  # Exit after the first batch for demonstration











'''
def evaluate_model(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            data = torch.cat(data, dim=0).to(device)
            data = min_max_normalize(data)
            reconstructed_x, z_mean, z_log_var = model(data)
            loss = vae_loss(reconstructed_x, data, z_mean, z_log_var)
            total_loss += loss.item()
    return total_loss / len(data_loader)



def min_max_normalize(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data.to(device)



### Training ###

# Assuming `data_loader` is your dataset loader providing batches of data
optimizer = optim.Adam(vae.parameters(), lr=0.01)
num_epochs = 100  # Set the number of epochs

vae=vae.to(device)

# Assuming that evaluate_model function returns the average validation loss for the entire validation set
def evaluate_model(model, loader):
    vae.eval()  # Set model to evaluation mode
    total_val_loss = 0
    with torch.no_grad():
        for data in tqdm(loader):
            data = torch.cat(data, dim=0).to(device)
            data = min_max_normalize(data)
            reconstructed_x, z_mean, z_log_var = model(data)
            val_loss = vae_loss(reconstructed_x, data, z_mean, z_log_var)
            total_val_loss += val_loss.item()
    return total_val_loss / len(loader)

# Initialize best validation loss
best_val_loss = float('inf')

vae.train()  # Ensure model is in training mode
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, data in enumerate(tqdm(train_loader)):
        data = torch.cat(data, dim=0).to(device)
        data = min_max_normalize(data)

        optimizer.zero_grad()
        reconstructed_x, z_mean, z_log_var = vae(data)
        loss = vae_loss(reconstructed_x, data, z_mean, z_log_var)
        loss.backward()

        assert not torch.isnan(loss).any(), "Loss contains nan values"
        assert not torch.isinf(loss).any(), "Loss contains inf values"

        for name, param in vae.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains nan values"
                assert not torch.isinf(param.grad).any(), f"Gradient for {name} contains inf values"

        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    val_loss = evaluate_model(vae, val_loader)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

    # Checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(vae.state_dict(), 'best_model.pth')  # Save the best model
        print(f'Checkpoint saved at epoch {epoch+1}')



# Assuming index2word is already defined as shown in your example
index2word = {v: k for k, v in tokenizer.vocab.items()}
index2word[0] = 'pad'  # Assuming 'pad' is your padding token


# Define the clean_text function
def clean_text(tokens, index2word):
    filtered_text = []
    for idx in tokens:
        token = index2word.get(idx.item(), '<UNK>')
        # Filter out padding and special tokens, adjust conditions as necessary
        if token not in ['<UNK>', 'pad', '[CLS]', '[SEP]']:
            filtered_text.append(token)
    return ' '.join(filtered_text)



# Test on a validation sentence
sent_idx = 2
# Assuming data_val is a PyTorch Tensor of appropriate shape and device
input_data = val_data[sent_idx:sent_idx+1].to(device) 

#print("input data", input_data)
#print("vae parameters", vae.parameters())

with torch.no_grad():
    vae.eval()  # Ensure the model is in evaluation mode
    logits, _, _ = vae(input_data) 
    probabilities = torch.softmax(logits, dim=-1)
    reconstructed_indexes = torch.argmax(probabilities, dim=-1)

    if reconstructed_indexes.nelement() > 1:
        reconstructed_indexes = reconstructed_indexes.squeeze()

    if reconstructed_indexes.dim() == 0:
        reconstructed_indexes = reconstructed_indexes.unsqueeze(0)

# Convert indices to words using clean_text
reconstructed_text = clean_text(reconstructed_indexes, index2word)
print('Reconstructed:', reconstructed_text)

# Getting the original sentence back from indices using clean_text
original_text = clean_text(input_data.squeeze(), index2word)
print('Original:', original_text)

'''