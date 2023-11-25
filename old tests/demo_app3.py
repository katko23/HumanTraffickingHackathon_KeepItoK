import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch

# Load data
negative_file = 'negative.txt'
positive_file = 'positive.txt'

with open(positive_file, 'r') as file:
    positive_examples = file.read().splitlines()

with open(negative_file, 'r') as file:
    negative_examples = file.read().splitlines()

# Combine positive and negative examples
texts = positive_examples + negative_examples
labels = [1] * len(positive_examples) + [0] * len(negative_examples)

# Create DataFrame
data = pd.DataFrame({"text": texts, "label": labels})

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize and encode the training data
train_encodings = tokenizer(train_data['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
train_labels = torch.tensor(train_data['label'].tolist())

# Tokenize and encode the test data
test_encodings = tokenizer(test_data['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
test_labels = torch.tensor(test_data['label'].tolist())

# Create DataLoader for training data
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Train the model
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, label = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=label)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Create DataLoader for test data
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Evaluate the model on the test set
model.eval()
predictions = []

for batch in test_loader:
    input_ids, attention_mask, _ = batch
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions.extend(torch.argmax(logits, dim=1).tolist())

# Calculate accuracy
accuracy = accuracy_score(test_data['label'], predictions)
print(f"Accuracy: {accuracy}")
