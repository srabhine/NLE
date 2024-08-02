import os
os.environ['KERAS_BACKEND'] = 'torch'

import keras
from keras import layers

import numpy as np
import math

import sys

import fidle
import pandas as pd
from torch.utils.data import DataLoader

import time
import torch
from transformers import BertTokenizer, BertModel
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

# Load data
train_data = pd.read_csv('/dm4i/work/towards.explainable.nlp/data/pcmag/train.csv')
val_data = pd.read_csv('/dm4i/work/towards.explainable.nlp/data/pcmag/valid.csv')
test_data = pd.read_csv('/dm4i/work/towards.explainable.nlp/data/pcmag/test.csv')

# Prepare the combined text columns
train_data['combined_text'] = train_data[['review_text', 'positive_comment', 'negative_comment', 'neural_comment']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
val_data['combined_text'] = val_data[['review_text', 'positive_comment', 'negative_comment', 'neural_comment']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
test_data['combined_text'] = test_data[['review_text', 'positive_comment', 'negative_comment', 'neural_comment']].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

def get_bert_embedding(text, window_size=512, overlap=256):
    # Ensure the text is within the tokenizer's capacity
    input_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512 * 100)
    input_ids = torch.tensor(input_ids).to(device)  # Move tensor to GPU right after creation
    
    # Sliding window approach
    start = 0
    embeddings = []
    while start + window_size <= len(input_ids):
        segment = input_ids[start:start + window_size].unsqueeze(0).to(device)
        with torch.no_grad():
            output = bert_model(segment)[0][:, 0, :]  # Keep on GPU
        embeddings.append(output)
        start += (window_size - overlap)

    # Handle the last segment if it hasn't been fully captured
    if start < len(input_ids):
        segment = input_ids[-window_size:].unsqueeze(0).to(device)
        with torch.no_grad():
            output = bert_model(segment)[0][:, 0, :]  # Keep on GPU
        embeddings.append(output)

    # Aggregate embeddings by averaging (on GPU)
    if embeddings:
        stacked_embeddings = torch.stack(embeddings)  # This will create a new tensor stacking the list of tensors
        return torch.mean(stacked_embeddings, dim=0)
    else:
        return torch.zeros((1, 768), device=device)  # Return a zero array on GPU if no embeddings were produced

# Extract embeddings
train_embeddings = [get_bert_embedding(text) for text in tqdm(train_data['combined_text'])]
val_embeddings = [get_bert_embedding(text) for text in tqdm(val_data['combined_text'])]
test_embeddings = [get_bert_embedding(text) for text in tqdm(test_data['combined_text'])]

#print("train_embeddings", train_embeddings)

# Other model configuration
NB_WORDS = len(tokenizer.vocab) + 1  # +1 for zero padding
EMBEDDING_DIM = 768
MAX_LEN = 512  # Set an appropriate maximum sequence length

# Create input ids and attention masks
def create_input_ids_and_attention_masks(data):
    input_ids = []
    attention_masks = []
    for text in data:
        encoded_dict = tokenizer.encode_plus(
            text,  # Text input
            add_special_tokens=True,
            max_length=MAX_LEN,  # Ensure MAX_LEN is appropriately defined
            padding='max_length',  # Pad to a consistent length
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids_tensor = encoded_dict['input_ids']
        attention_masks_tensor = encoded_dict['attention_mask']

        # Debugging output
        #print(f"Shape of input_ids: {input_ids_tensor.shape}")
        #print(f"Shape of attention_mask: {attention_masks_tensor.shape}")

        input_ids.append(input_ids_tensor.transpose(0,1))
        attention_masks.append(attention_masks_tensor.transpose(0,1))

    # Concatenate all tensors along the batch dimension
    concatenated_input_ids = torch.cat(input_ids, dim=0)
    concatenated_attention_masks = torch.cat(attention_masks, dim=0)

    # More debugging output
    #print(f"Final shape of concatenated input_ids: {concatenated_input_ids.shape}")
    #print(f"Final shape of concatenated attention_masks: {concatenated_attention_masks.shape}")

    return concatenated_input_ids, concatenated_attention_masks


train_input_ids, train_attention_masks = create_input_ids_and_attention_masks(train_data['combined_text'])
val_input_ids, val_attention_masks = create_input_ids_and_attention_masks(val_data['combined_text'])
test_input_ids, test_attention_masks = create_input_ids_and_attention_masks(test_data['combined_text'])

# Data loaders
batch_size = 64
train_loader = DataLoader(TensorDataset(train_input_ids, train_attention_masks), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(TensorDataset(val_input_ids, val_attention_masks), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(TensorDataset(test_input_ids, test_attention_masks), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

total_batches = len(train_loader)
print("Total batches:", total_batches)

total_data_points = len(train_data['combined_text'])
print("Total data points:", total_data_points)
print("Expected batches:", total_data_points / batch_size)

#print("test loader", test_loader)



### VAE Model ###

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerVAEEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, num_layers, num_heads):
        super(TransformerVAEEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_mu = nn.Linear(embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(embedding_dim, latent_dim)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        encoded_src = self.transformer_encoder(src)
        mu = self.fc_mu(encoded_src.mean(dim=1))
        logvar = self.fc_logvar(encoded_src.mean(dim=1))
        return mu, logvar, encoded_src
    

class TransformerVAEDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, num_layers, num_heads):
        super(TransformerVAEDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        decoder_layers = nn.TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

        # Additional layer to transform z from latent space to embedding dimension space
        self.latent_to_embedding = nn.Linear(latent_dim, embedding_dim)

    def forward(self, z, memory):
        # Transform latent vector z to the embedding dimension
        z = self.latent_to_embedding(z).unsqueeze(0)  # Unsqueeze to add the sequence length dimension
        z = self.pos_encoder(z)
        memory = memory.transpose(0, 1)
        #print("z shape:", z.shape)
        #print("memory shape:", memory.shape)
        output = self.transformer_decoder(z, memory)
        logits = self.fc_out(output)
        return logits

class TransformerVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, num_layers, num_heads):
        super(TransformerVAE, self).__init__()
        self.encoder = TransformerVAEEncoder(vocab_size, embedding_dim, hidden_dim, latent_dim, num_layers, num_heads)
        self.decoder = TransformerVAEDecoder(vocab_size, embedding_dim, hidden_dim, latent_dim, num_layers, num_heads)

    def forward(self, src, z=None):
        mu, logvar, memory = self.encoder(src)
        z = self.reparameterize(mu, logvar) if z is None else z
        return self.decoder(z, memory), mu, logvar
    

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps




    def vae_loss(recon_x, x, mu, logvar, kl_weight=0.01):
        # recon_x is [1, 30523, 64] - should ideally be [64, 30523]
        # Change recon_x to [64, 30523] for per-timestep classification
        recon_x = recon_x.squeeze(0).transpose(0, 1)  # Changes recon_x to [30523, 64]
        recon_x = recon_x.unsqueeze(2)  # Changes recon_x to [30523, 64, 1]
        recon_x = recon_x.transpose(0, 1)  # Final recon_x is [64, 30523, 1]

        # Flatten x if necessary
        x = x.view(-1)  # Change x from [64, 1] to [64]

        # Check shapes
        #print("recon_x shape after transpose and permute:", recon_x.shape)
        #print("x shape after view:", x.shape)

        # Calculate reconstruction loss
        recon_loss = nn.CrossEntropyLoss()(recon_x.squeeze(2), x)  # Now x and recon_x should be compatible

        # Calculate KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total VAE loss
        return recon_loss + kl_weight * kl_loss
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint_3layers.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss




# Example initialization
model = TransformerVAE(vocab_size=NB_WORDS, embedding_dim=300, hidden_dim=512, latent_dim=64, num_layers=3, num_heads=20)

model.load_state_dict(torch.load('checkpoint_3layers.pth', map_location=device))

model.to(device)

'''
def validate(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, _ = data  # Assuming only inputs are passed and no actual 'targets'
            inputs = inputs.to(device)
            
            # Ensure all outputs from the model are captured
            outputs, mu, logvar = model(inputs)
            
            # The targets for VAE are typically the inputs themselves for reconstruction
            loss = loss_fn(outputs, inputs, mu, logvar)
            val_loss += loss.item()

    return val_loss / len(val_loader)



optimizer = optim.Adam(model.parameters(), lr=0.01)

early_stopping = EarlyStopping(patience=3, verbose=True)


optimizer = optim.Adam(model.parameters(), lr=0.0001)



# Training loop

epochs = 10  # Number of epochs, adjust as necessary
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

    for batch_idx, (input_ids, attention_mask) in progress_bar:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)  # Ensure attention_mask is also sent to device if used in the model

        optimizer.zero_grad()
        outputs = model(input_ids)  # Ensure that attention_mask is used in the model if necessary
        recon_batch, mu, logvar = outputs

        loss = TransformerVAE.vae_loss(recon_batch, input_ids, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % fit_verbosity == 0:
            progress_bar.set_description(f'Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item()/len(input_ids):.6f}')

    avg_train_loss = total_loss / len(train_loader)
    print(f'====> Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}')

    # Validate the model
    avg_val_loss = validate(model, val_loader, TransformerVAE.vae_loss)
    print(f'====> Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f}')

    # Check early stopping criterion
    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

print("Training completed successfully.")




def tokens_to_text(token_ids, index2word):
    """ Convert a list of token IDs to their corresponding words, filtering out special tokens. """
    return ' '.join([index2word.get(token.item(), '<UNK>') for token in token_ids if index2word.get(token.item(), '<UNK>') not in ['<UNK>', 'pad', '[CLS]', '[SEP]']])



def tokens_to_text(token_ids, tokenizer):
    """ Convert a list of token IDs to their corresponding words, filtering out special tokens. """
    # Convert the token IDs to a PyTorch tensor if not already one
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids)
    
    # Use the tokenizer to convert the token IDs to text, skipping special tokens like padding, etc.
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text

'''

model.eval()

sent_idx = 20

### BERT Score ###


from bert_score import BERTScorer

def evaluate_bert_scores(model, data_loader, tokenizer, device):
    model.eval()
    generated_texts = []
    reference_texts = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids, attention_masks = batch
            input_ids, attention_masks = input_ids.to(device), attention_masks.to(device)
            outputs, _, _ = model(input_ids)
            _, predicted_ids = torch.max(outputs, dim=2)
            generated_texts += tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            reference_texts += tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    scorer = BERTScorer(lang='en', model_type='bert-base-uncased')

    bert_score= []

    for generated_text, reference_text in zip(generated_texts, reference_texts):
        if generated_text.strip() and reference_text.strip():
            # Scoring individual pairs
            score = scorer.score([reference_text], [generated_text])
            bert_score.append(score)
            print(f"Score for reference and generated text: {score}")
        else:
            print(f"Empty or invalid texts detected, skipping: '{reference_text}' | '{generated_text}'")

    
    mean_bert_score = np.mean(bert_score)
    print(f"Mean BERT score: {mean_bert_score:.4f}")




evaluate_bert_scores(model, val_loader, tokenizer, device)


def tokens_to_text(token_ids, index2word):
    """ Convert a list of token IDs to their corresponding words, filtering out special tokens. """
    return ' '.join([index2word.get(token.item(), '<UNK>') for token in token_ids if index2word.get(token.item(), '<UNK>') not in ['<UNK>', 'pad', '[CLS]', '[SEP]']])



def tokens_to_text(token_ids, tokenizer):
    """ Convert a list of token IDs to their corresponding words, filtering out special tokens. """
    # Convert the token IDs to a PyTorch tensor if not already one
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids)
    
    # Use the tokenizer to convert the token IDs to text, skipping special tokens like padding, etc.
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text



with torch.no_grad():
    for i, (input_ids, attention_mask) in enumerate(val_loader):  # Unpack the data properly
        if i == 0:  # Just taking the first batch for simplicity
            

            input_ids = input_ids.to(device)  # Send input data to the device
            print("input ids shape", input_ids.shape)
            # Forward pass through the model
            logits, _, _ = model(input_ids)
            print("shape of logits:",logits.shape)
            probabilities = torch.softmax(logits, dim=-1)
            reconstructed_indexes = torch.argmax(probabilities, dim=-1)
            print("reconstructed indexes shape", reconstructed_indexes)
            
            # Handle dimensions if necessary
            if reconstructed_indexes.dim() > 2:
                reconstructed_indexes = reconstructed_indexes.squeeze()

            # Convert the indices of the first sentence to text
            reconstructed_text = tokens_to_text(reconstructed_indexes[sent_idx].cpu().numpy(), tokenizer)
            original_text = tokens_to_text(input_ids[sent_idx].cpu().numpy(), tokenizer)

            print('Reconstructed:', reconstructed_text)
            print('Original:', original_text)
            break  # Exit after the first batch for demonstration
            

            

'''
#### 2ND  try ####

with torch.no_grad():
    for i, (input_ids, attention_mask) in enumerate(val_loader):
        if i == 0:  # Just taking the first batch for simplicity
            input_ids = input_ids.to(device)  # Send input data to the device

            # Forward pass through the model
            logits, _, _ = model(input_ids)
            probabilities = torch.softmax(logits, dim=-1)
            reconstructed_indexes = torch.argmax(probabilities, dim=-1)
            
            # Adjust dimension if necessary and ensure access within bounds
            reconstructed_indexes = reconstructed_indexes.squeeze(0)  # This removes the first dimension, making it [64]
            
            # Check if sent_idx is within the range of the batch size
            if sent_idx >= input_ids.shape[0]:
                print(f"Index {sent_idx} is out of bounds for the current batch. Using index 0 instead.")
                sent_idx = 0  # Default to the first index if out of range

            # Convert the indices of the chosen sentence to text
            print("Token IDs:", reconstructed_indexes[sent_idx].cpu().numpy())
            reconstructed_text = tokens_to_text(reconstructed_indexes[sent_idx].cpu().numpy(), tokenizer)
            original_text = tokens_to_text(input_ids[sent_idx].cpu().numpy(), tokenizer)

            print('Reconstructed:', reconstructed_text)
            print('Original:', original_text)
            break  # Exit after the first batch for demonstration



import torch

def generate_sequence_beam_search(model, tokenizer, input_ids, beam_size=3, max_length=50):
    model.eval()
    with torch.no_grad():
        # Initialize beams as a list of tokens, with log probabilities
        beams = [(input_ids, 0)]  # (current_sequence, current_log_prob)
        
        for _ in range(max_length):
            new_beams = []
            for beam in beams:
                seq, log_prob = beam
                if seq[0, -1].item() == tokenizer.eos_token_id:
                    new_beams.append(beam)  # Keep if EOS token was generated
                    continue
                
                # Get probabilities from the model output
                logits = model(seq)[0][:, -1, :]  # Only consider the last output
                probs = torch.softmax(logits, dim=-1)
                
                # Get top beam_size token probabilities
                top_probs, top_ix = probs.topk(beam_size)
                
                # Append new sequences to the beams
                for i in range(beam_size):
                    next_seq = torch.cat([seq, top_ix[:, i].unsqueeze(-1)], dim=1)
                    next_log_prob = log_prob + top_probs[:, i].log().item()  # Add log probabilities for numerical stability
                    new_beams.append((next_seq, next_log_prob))
            
            # Sort by probability and select top beam_size sequences
            new_beams.sort(key=lambda b: b[1], reverse=True)
            beams = new_beams[:beam_size]
        
        # Choose the sequence with the highest probability
        best_seq = beams[0][0]
        return best_seq

# Example usage
prompt_text = "The weather is nice today"
encoded_input = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
generated_sequence = generate_sequence_beam_search(model, tokenizer, encoded_input, beam_size=5)

# Decode generated sequence back to text
generated_text = tokenizer.decode(generated_sequence[0].cpu().numpy(), skip_special_tokens=True)
print("Generated Text:", generated_text)



encoder = TransformerVAEEncoder(vocab_size=NB_WORDS, embedding_dim=300, hidden_dim=512, latent_dim=64, num_layers=6, num_heads=10)
decoder = TransformerVAEDecoder(vocab_size=NB_WORDS, embedding_dim=300, hidden_dim=512, latent_dim=64, num_layers=6, num_heads=10)

encoder.to(device)
decoder.to(device)


def sent_parse(sentence, max_len):
    sequence = tokenizer.encode(sentence, add_special_tokens=False)
    sequence = torch.tensor(sequence)
    padded_sent = pad(sequence, (0, max_len - sequence.shape[0]), mode='constant', value=0)
    return padded_sent.unsqueeze(0)  # Adding batch dimension

import torch

def find_similar_encoding(sent_vect, sent_encoded):
    all_cosine = torch.nn.functional.cosine_similarity(sent_vect, sent_encoded, dim=1)
    _, indices = torch.topk(all_cosine, k=3)
    return sent_encoded[indices[1]]  # Skipping the first one as it's the input itself



def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = torch.linspace(0, 1, steps=num)
    hom_sample = [point_one + s * dist_vec for s in sample]
    return hom_sample



def print_latent_sentence(sent_vect, generator, max_len, NB_WORDS, index2word):
    sent_vect = sent_vect.unsqueeze(0)  # Adding batch dimension
    sent_reconstructed = generator(sent_vect)
    _, reconstructed_indexes = torch.max(sent_reconstructed, dim=2)
    word_list = [index2word[idx.item()] for idx in reconstructed_indexes[0] if idx.item() in index2word]
    print(' '.join(word_list))



def new_sents_interp(sent1, sent2, n, MAX_SEQUENCE_LENGTH, encoder, generator, max_len, NB_WORDS, index2word):
    tok_sent1 = sent_parse(sent1, MAX_SEQUENCE_LENGTH)
    tok_sent2 = sent_parse(sent2, MAX_SEQUENCE_LENGTH)
    enc_sent1 = encoder(tok_sent1)
    enc_sent2 = encoder(tok_sent2)
    test_hom = shortest_homology(enc_sent1, enc_sent2, n)
    for point in test_hom:
        print_latent_sentence(point, generator, max_len, NB_WORDS, index2word)



######## SUITE ######

# Tokenization and Padding
def sent_parse(sentence, max_len):
    # Assuming `tokenizer` is already a pretrained tokenizer compatible with PyTorch
    sequence = tokenizer.encode(sentence, add_special_tokens=False, return_tensors='pt')
    padded_sent = torch.nn.functional.pad(sequence, (0, max_len - sequence.size(1)), value=tokenizer.pad_token_id)
    return padded_sent

# Finding Similar Encodings
def find_similar_encoding(sent_vect, sent_encoded):
    # Using cosine similarity to find the most similar encoding
    all_cosine = torch.nn.functional.cosine_similarity(sent_vect, sent_encoded, dim=1)
    _, indices = torch.topk(all_cosine, k=3)
    return sent_encoded[indices[1]]  # Skipping the first one as it's the input itself

# Linear Interpolation
def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = torch.linspace(0, 1, steps=num, device=point_one.device)
    return [point_one + s * dist_vec for s in sample]


def print_latent_sentence(sent_vect, generator, tokenizer):
    """Generate and print a sentence from a latent vector using the generator model."""
    sent_vect = sent_vect.unsqueeze(0)  # Adding batch dimension if not present
    sent_reconstructed = generator(sent_vect)
    _, reconstructed_indexes = torch.max(sent_reconstructed, dim=2)
    
    # Decode token IDs back to text using the tokenizer
    decoded_sentence = tokenizer.decode(reconstructed_indexes[0], skip_special_tokens=True)
    print(decoded_sentence)


# Example code execution
sentence1 = 'gogogo where can i find a bad restaurant endend'
MAX_SEQUENCE_LENGTH = 30  # This should be defined according to your model's requirements

mysent = sent_parse(sentence1, MAX_SEQUENCE_LENGTH + 2)
mysent_encoded = encoder
print_latent_sentence(mysent_encoded,decoder, tokenizer)
print_latent_sentence(find_similar_encoding(mysent_encoded, mysent_encoded))

sentence2 = 'gogogo where can i find an extremely good restaurant endend'
mysent2 = sent_parse(sentence2, MAX_SEQUENCE_LENGTH + 2)
mysent_encoded2 = TransformerVAEEncoder(mysent2)
print_latent_sentence(mysent_encoded2)
print_latent_sentence(find_similar_encoding(mysent_encoded2, mysent_encoded2))
print('-----------------')

new_sents_interp(sentence1, sentence2, 5)

# Interpolation and Sentence Generation
def new_sents_interp(sent1, sent2, n):
    tok_sent1 = sent_parse(sent1, MAX_SEQUENCE_LENGTH + 2)
    tok_sent2 = sent_parse(sent2, MAX_SEQUENCE_LENGTH + 2)
    enc_sent1 = TransformerVAEEncoder(tok_sent1)
    enc_sent2 = TransformerVAEEncoder(tok_sent2)
    test_hom = shortest_homology(enc_sent1.squeeze(0), enc_sent2.squeeze(0), n)
    for point in test_hom:
        print_latent_sentence(point)


'''