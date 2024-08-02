import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import re
from torch.utils.data import TensorDataset
from tqdm import tqdm
import torch.optim as optim
import os
import optuna

#os.environ['CUDA_LAUNCH_BLOCKING'] = 1

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#tokenizer.save_pretrained("tokenizer")
#print("tokenizer saved")
#tokenizer = BertTokenizer.from_pretrained("/gpfsgaia/home/d20584/tokenizer")



#tokenizer initialisation, and vocab size intialization

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size


#train, val, test dataframes

dftrain=pd.read_csv('expanded_train.csv')
dfval=pd.read_csv('expanded_val.csv')
dftest=pd.read_csv('expanded_test.csv')


#text cleaning

html_tag_re = re.compile(r'<.*?>')
url_re = re.compile(r'http\S+|www\S+|https\S+')
newline_re = re.compile(r'\n')
special_char_re = re.compile(r'[^A-Za-z0-9\s]')

# Function to clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Lowercase the text
    return text

# Apply the clean_text function to both 'review_text' and 'comment' columns
dftrain['review_text'] = dftrain['review_text'].apply(clean_text)
dftrain['comment'] = dftrain['comment'].apply(clean_text)
dfval['review_text'] = dftrain['review_text'].apply(clean_text)
dfval['comment'] = dftrain['comment'].apply(clean_text)
dftest['review_text'] = dftrain['review_text'].apply(clean_text)
dftest['comment'] = dftrain['comment'].apply(clean_text)


'''

# Function to encode two columns
def encode_data(data, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []
    for review, comment in tqdm(zip(data['review_text'], data['comment']), total=len(data)):
        # Tokenize the review text
        encoded_review = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        # Tokenize the comment text
        encoded_comment = tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        # Concatenate the input IDs and attention masks
        combined_input_ids = torch.cat((encoded_review['input_ids'], encoded_comment['input_ids']), dim=1)
        combined_attention_masks = torch.cat((encoded_review['attention_mask'], encoded_comment['attention_mask']), dim=1)
        
        input_ids.append(combined_input_ids)
        attention_masks.append(combined_attention_masks)
    
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

# Encode the data
train_input_ids, train_attention_masks = encode_data(dftrain, tokenizer)
val_input_ids, val_attention_masks = encode_data(dfval, tokenizer)
test_input_ids, test_attention_masks = encode_data(dftest, tokenizer)

torch.save(train_input_ids, 'train_input_ids.pt')
torch.save(train_attention_masks, 'train_attention_masks.pt')
torch.save(val_input_ids, 'val_input_ids.pt')
torch.save(val_attention_masks, 'val_attention_masks.pt')
torch.save(test_input_ids, 'test_input_ids.pt')
torch.save(test_attention_masks, 'test_attention_masks.pt')

'''


#load the ids and the mask that i saved earlier
train_input_ids = torch.load('train_input_ids.pt')
train_attention_masks = torch.load('train_attention_masks.pt')
val_input_ids = torch.load('val_input_ids.pt')
val_attention_masks = torch.load('val_attention_masks.pt')
test_input_ids = torch.load('test_input_ids.pt')
test_attention_masks = torch.load('test_attention_masks.pt')


#Map the labels to values 
train_labels = torch.tensor(dftrain['label'].map({'Neg': 0, 'Neu': 1, 'Pos': 2}).values, dtype=torch.long)
val_labels = torch.tensor(dfval['label'].map({'Neg': 0, 'Neu': 1, 'Pos': 2}).values, dtype=torch.long)
test_labels = torch.tensor(dftest['label'].map({'Neg': 0, 'Neu': 1, 'Pos': 2}).values, dtype=torch.long)


#created variables for the ratings
train_ratings = torch.tensor(dftrain['overall_rating'].values, dtype=torch.long).to(device)
val_ratings = torch.tensor(dfval['overall_rating'].values, dtype=torch.long).to(device)
test_ratings = torch.tensor(dftest['overall_rating'].values, dtype=torch.long).to(device)

'''

print("Training ratings range:", train_ratings.min().item(), train_ratings.max().item())
print("Validation ratings range:", val_ratings.min().item(), val_ratings.max().item())
print("Test ratings range:", test_ratings.min().item(), test_ratings.max().item())

print("Unique training ratings:", torch.unique(train_ratings).cpu().numpy())
print("Unique validation ratings:", torch.unique(val_ratings).cpu().numpy())
print("Unique test ratings:", torch.unique(test_ratings).cpu().numpy())

'''



train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels, train_ratings)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels, val_ratings)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels, test_ratings)


train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

'''
train_dataset_comments=TensorDataset(train_input_ids, train_attention_masks, train_ratings)
val_dataset_comments=TensorDataset(train_input_ids, train_attention_masks, train_ratings)
test_dataset_comments=TensorDataset(train_input_ids, train_attention_masks, train_ratings)

train_loader_comment = DataLoader(train_dataset_comments, batch_size=128, shuffle=True)
val_loader_comment = DataLoader(val_dataset_comments, batch_size=128, shuffle=False)
test_loader_comment = DataLoader(test_dataset_comments, batch_size=128, shuffle=False)
'''




#Encoding the column "comment" 
def encode_data_comment(data, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []
    for comment in tqdm(data['comment'], total=len(data)):
        encoded = tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    #print(f"Encoded input_ids shape: {input_ids.shape}")
    #print(f"Encoded attention_masks shape: {attention_masks.shape}")

    return input_ids, attention_masks

# Encode the data
train_input_ids_comment, train_attention_masks_comment = encode_data_comment(dftrain, tokenizer)
val_input_ids_comment, val_attention_masks_comment = encode_data_comment(dfval, tokenizer)
test_input_ids_comment, test_attention_masks_comment = encode_data_comment(dftest, tokenizer)

train_dataset_comment = TensorDataset(train_input_ids_comment, train_attention_masks_comment, train_input_ids_comment, train_attention_masks_comment)
val_dataset_comment = TensorDataset(val_input_ids_comment, val_attention_masks_comment, val_input_ids_comment, val_attention_masks_comment)
test_dataset_comment = TensorDataset(test_input_ids_comment, test_attention_masks_comment, test_input_ids_comment, test_attention_masks_comment)

train_loader_comment = DataLoader(train_dataset_comment, batch_size=16, shuffle=True)
val_loader_comment = DataLoader(val_dataset_comment, batch_size=16, shuffle=False)
test_loader_comment = DataLoader(test_dataset_comment, batch_size=16, shuffle=False)



#Training of the classifier on our data


'''
def train_classifier(model, train_dataloader, val_dataloader, epochs=10, learning_rate=1e-4, save_path='bert_classifier.pth', patience=3):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()

    best_val_loss = float('inf')
    no_improvement = 0

    for epoch in range(epochs):
        total_train_loss = 0
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            input_ids, attention_masks, ratings = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            loss = criterion(logits, ratings)  # Using ratings as labels for the loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}")

        # Evaluate on the validation set
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}"):
                input_ids, attention_masks, ratings = batch
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                ratings = ratings.to(device)

                outputs = model(input_ids, attention_mask=attention_masks)
                logits = outputs.logits
                loss = criterion(logits, ratings)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss}")

        # Early stopping based on the validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"Model weights saved to {save_path}")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    print("Training complete")


train_classifier(model, train_loader_comment, val_loader_comment)




from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertConfig

model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(model_name, hidden_dropout_prob=0.5, num_labels=9) # added the hidden dropout because of the overfitting and num_labels=9 because of the dispersion of the overall ratings values 
model = BertForSequenceClassification.from_pretrained(model_name, config=config).to(device)

model.load_state_dict(torch.load('best_bert_classifier.pth'))

# Load the pretrained classifier weights
pretrained_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=9).to(device)
pretrained_classifier.load_state_dict(torch.load('best_bert_classifier.pth'))
pretrained_classifier.eval() 

'''


from transformers import BertTokenizer

#CVAE class

class CVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, num_labels, vocab_size, sequence_length, rating_classes, pretrained_classifier):
        super(CVAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.rating_classes = rating_classes
        self.pretrained_classifier = pretrained_classifier
        
        # Text encoder E
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        
        # Recognition Network
        self.recog_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.recog_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.recog_fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Explanation Generator G
        self.decoder_input_fc = nn.Linear(latent_dim, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8), num_layers=6
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        # BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        ve = outputs.last_hidden_state.mean(dim=1)
        return ve
    
    def recognition_network(self, ve):
        h = F.relu(self.recog_fc1(ve))
        mu = self.recog_fc_mu(h)
        logvar = self.recog_fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    
    
    def generate_explanation(self, z):
        h = self.decoder_input_fc(z)
        h = h.unsqueeze(1).repeat(1, self.sequence_length, 1)  # Repeat for sequence length
        memory = self.transformer_encoder(h)
        generated_explanation = self.fc_out(memory)

        # Debug prints
        #print(f"Generated explanation shape before view: {generated_explanation.shape}")
        expected_elements = h.size(0) * self.sequence_length * self.vocab_size
        #print(f"Expected elements: {expected_elements}, Actual elements: {generated_explanation.numel()}")

        generated_explanation = generated_explanation.view(-1, self.sequence_length, self.vocab_size)

        return generated_explanation
    

    
    
    def decode_explanation(self, logits):
        ###### torch.multinomial ####
        explanation_ids = torch.argmax(logits, dim=-1)
        explanations = [self.tokenizer.decode(ids.tolist(), skip_special_tokens=True) for ids in explanation_ids]
        return explanations
    

    
    def tokenize_explanation(self, explanation_logits):

        explanation_ids = torch.argmax(explanation_logits, dim=-1)

        #print("Generative explanations ids shape", explanation_ids.shape)

        if explanation_ids.dim() == 1:
            explanation_ids = explanation_ids.unsqueeze(0)
        
        explanations = [self.tokenizer.decode(ids.tolist(), skip_special_tokens=True) for ids in explanation_ids]
        tokenized = self.tokenizer(explanations, return_tensors='pt', padding=True, truncation=True, max_length=128)

        #print("Tokenized generative input_ids shape:", tokenized['input_ids'].shape)
        #print("Tokenized generative attention_mask shape:", tokenized['attention_mask'].shape)
        
        return tokenized['input_ids'].to(explanation_logits.device), tokenized['attention_mask'].to(explanation_logits.device)
    
    def classify_explanation(self, explanation_ids, attention_mask):
        explanation_ids = explanation_ids.to(self.pretrained_classifier.device)
        attention_mask = attention_mask.to(self.pretrained_classifier.device) 
        outputs = self.pretrained_classifier(explanation_ids, attention_mask=attention_mask)
        return F.softmax(outputs.logits, dim=1)
    

    def forward(self, input_ids, attention_mask, golden_input_ids, golden_attention_mask, sequence_length=256):
        ve = self.encode(input_ids, attention_mask)
        mu, logvar = self.recognition_network(ve)
        z = self.reparameterize(mu, logvar)

        # Generate explanations (logits)
        self.sequence_length = sequence_length
        generative_logits = self.generate_explanation(z)

        # Tokenize generative explanations
        generative_explanations = generative_logits.view(-1, self.vocab_size).argmax(dim=-1)
        generative_explanations = self.tokenizer.batch_decode(generative_explanations, skip_special_tokens=True)
        tokenized_generative = self.tokenizer(generative_explanations, return_tensors='pt', padding=True, truncation=True, max_length=sequence_length)

        # Tokenize generative explanations again
        generative_input_ids, generative_attention_mask = self.tokenize_explanation(tokenized_generative['input_ids'])

        # Classify generative explanations
        P_classified = self.classify_explanation(generative_input_ids, generative_attention_mask)

        # Handle golden explanations
        if golden_input_ids.dim() == 1:
            golden_input_ids = golden_input_ids.unsqueeze(0)

        golden_input_ids = golden_input_ids.tolist()  # Ensure the input is a list of token ids
        golden_explanations = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in golden_input_ids]
        tokenized_golden = self.tokenizer(golden_explanations, return_tensors='pt', padding=True, truncation=True, max_length=sequence_length)

        padded_golden_input_ids = tokenized_golden['input_ids'].to(input_ids.device)
        padded_golden_attention_mask = tokenized_golden['attention_mask'].to(input_ids.device)

        #print("padded golden input ids", padded_golden_input_ids.shape)
        #print("padded golden attention mask", padded_golden_attention_mask.shape)

        P_gold = self.classify_explanation(padded_golden_input_ids, padded_golden_attention_mask)

        #print("P_gold", P_gold.shape)

        P_pred = self.classify_explanation(input_ids, attention_mask)

        #print("P_pred", P_pred.shape)

        return P_classified, P_gold, P_pred, mu, logvar, generative_logits


# pretrained classifier instantiation
pretrained_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=9).to(device)
pretrained_classifier.load_state_dict(torch.load('best_bert_classifier.pth'))
pretrained_classifier.eval()

cvae = CVAE(hidden_dim=768, latent_dim=64, num_labels=3, vocab_size=30522, sequence_length=256, rating_classes=9, pretrained_classifier=pretrained_classifier).to(device)


########################################## Training the CVAE ################################################

from torch.cuda.amp import GradScaler, autocast

def explanation_factor_loss(P_classified, P_gold, P_pred):
    return torch.sum(torch.abs(P_classified - P_gold) + torch.abs(P_classified - P_pred))



def minimum_risk_training_loss(P_classified, P_gold, train_loader, semantic_distance, cvae, device):
    lmrt_loss = 0.0
    num_examples = 0

    # Iterate over the training loader to calculate the expected value of the semantic distance
    for batch in train_loader:
        input_ids, attention_mask, golden_input_ids, golden_attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        golden_input_ids = golden_input_ids.to(device)
        golden_attention_mask = golden_attention_mask.to(device)

        # Calculate the predicted overall results for the input examples in the batch
        P_pred_batch = cvae.classify_explanation(input_ids, attention_mask)

        # Calculate the semantic distance between the predicted overall results and the ground-truth overall results for each example in the batch
        batch_loss = semantic_distance(P_pred_batch, P_gold)

        # Sum the losses for all examples in the batch
        lmrt_loss += batch_loss.sum().item()
        num_examples += batch_loss.numel()

    # Calculate the expected value of the semantic distance over the training dataset
    lmrt_loss /= num_examples

    # Explanation factor loss
    ef_loss = explanation_factor_loss(P_classified, P_gold, P_pred_batch)

    return lmrt_loss * ef_loss



def semantic_distance(P_pred_batch, P_gold):
    # You can define your own semantic distance function. Here's an example using mean squared error.
    return torch.mean((P_pred_batch - P_gold) ** 2, dim=1)


    

def combined_loss_function(P_classified, P_gold, P_pred, P_true, mu, logvar, generative_logits, input_ids, train_loader, device):
    # Explanation generation loss (CrossEntropyLoss)
    explanation_loss = nn.CrossEntropyLoss()(generative_logits.view(-1, generative_logits.size(-1)), input_ids.view(-1))

    # Minimum risk training loss
    mrt_loss = minimum_risk_training_loss(P_classified, P_gold, train_loader, semantic_distance, cvae, device)

    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

    # CVAE loss
    cvae_loss = explanation_loss + kld_loss

    # Final loss
    final_loss = cvae_loss + mrt_loss

    return final_loss




#Hyperparameters
latent_dim = 64
learning_rate = 1e-3

cvae = CVAE(hidden_dim=768, latent_dim=latent_dim, num_labels=3, vocab_size=30522, sequence_length=256, rating_classes=6, pretrained_classifier=pretrained_classifier).to(device)
    
optimizer = optim.Adam(cvae.parameters(), lr=learning_rate)
#Added the GradScaler() for training optimization 
scaler = GradScaler()
cvae.train()

best_val_loss = float('inf')

for epoch in range(1):  
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids, attention_mask, golden_input_ids, golden_attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        golden_input_ids = golden_input_ids.to(device)
        golden_attention_mask = golden_attention_mask.to(device)
        
        optimizer.zero_grad()
        with autocast():
            P_classified, P_gold, P_pred, mu, logvar, generative_explanations = cvae(input_ids, attention_mask, golden_input_ids, golden_attention_mask)
            loss = combined_loss_function(P_classified, P_gold, P_pred, P_gold, mu, logvar, generative_explanations, input_ids, train_loader, device)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    

# Evaluate on the validation set
    cvae.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            input_ids, attention_mask, golden_input_ids, golden_attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            golden_input_ids = golden_input_ids.to(device)
            golden_attention_mask = golden_attention_mask.to(device)
            
            P_classified, P_gold, P_pred, mu, logvar, generative_explanations = cvae(input_ids, attention_mask, golden_input_ids, golden_attention_mask)
            loss = combined_loss_function(P_classified, P_gold, P_pred, P_gold, mu, logvar, generative_explanations, input_ids, val_loader, device)
            
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}, Training Loss: {avg_loss}, Validation Loss: {avg_val_loss}")

# Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(cvae.state_dict(), "model_best_gaia.pth")




# Optuna opmization to find the best hp

'''
# Run the optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)

'''



# Test the generation on the test data

'''

##### Test on data ######

def generate_explanations(cvae, tokenizer, input_ids, attention_mask, golden_input_ids, golden_attention_mask, labels):
    cvae.eval()
    
    with torch.no_grad():
        P_classified, P_gold, P_pred, _, _, generative_explanations = cvae(input_ids, attention_mask, golden_input_ids, golden_attention_mask)

    explanations = tokenizer.batch_decode(generative_explanations.argmax(dim=-1), skip_special_tokens=True)
    golden_explanations = tokenizer.batch_decode(golden_input_ids, skip_special_tokens=True)

    results = []
    for i in range(len(input_ids)):
        explanation = explanations[i]
        golden_explanation = golden_explanations[i]
        label = labels[i].item()
        
        if label == 0:
            label_str = "Negative"
        elif label == 1:
            label_str = "Neutral"
        else:
            label_str = "Positive"

        result = {
            "label": label_str,
            "generated": explanation,
            "golden": golden_explanation
        }
        results.append(result)
    
    return results

    

# Load your data and prepare DataLoader
df = pd.read_csv('/mnt/data/your_test_data.csv')
input_ids = torch.tensor(df['input_ids'].tolist())
attention_mask = torch.tensor(df['attention_mask'].tolist())
golden_input_ids = torch.tensor(df['golden_input_ids'].tolist())
golden_attention_mask = torch.tensor(df['golden_attention_mask'].tolist())
labels = torch.tensor(df['labels'].tolist())

dataset = TensorDataset(input_ids, attention_mask, golden_input_ids, golden_attention_mask, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load your trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
cvae = CVAE(hidden_dim=768, latent_dim=64, num_labels=3, vocab_size=30522, sequence_length=256, rating_classes=6, pretrained_classifier=pretrained_classifier)
cvae.load_state_dict(torch.load('/mnt/data/trained_cvae_model.pth'))
cvae.to(device)
cvae.eval()

# Generate explanations for all data in the DataLoader
all_results = []
for batch in dataloader:
    input_ids, attention_mask, golden_input_ids, golden_attention_mask, labels = [b.to(device) for b in batch]
    results = generate_explanations(cvae, tokenizer, input_ids, attention_mask, golden_input_ids, golden_attention_mask, labels)
    all_results.extend(results)

# Extract one example of each label
negative_example = next(result for result in all_results if result['label'] == 'Negative')
neutral_example = next(result for result in all_results if result['label'] == 'Neutral')
positive_example = next(result for result in all_results if result['label'] == 'Positive')

# Print the results
print("Negative Example:")
print(f"Generated Explanation: {negative_example['generated']}")
print(f"Golden Explanation: {negative_example['golden']}")
print()

print("Neutral Example:")
print(f"Generated Explanation: {neutral_example['generated']}")
print(f"Golden Explanation: {neutral_example['golden']}")
print()

print("Positive Example:")
print(f"Generated Explanation: {positive_example['generated']}")
print(f"Golden Explanation: {positive_example['golden']}")
print()

'''



























