import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score
import parser

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load your labeled dataset with messages and labels (0 for not related, 1 for related)
# Replace 'your_dataset.csv' with the path to your dataset
dataset = pd.read_csv('dataset.csv')

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

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
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
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

# Make predictions on new messages
new_messages = [
    "I'm scared and I can't leave.",
    "I was promised a good job, but now I'm being forced to work in terrible conditions.",
    "I don't have control over my own life.",
    "but I can write a simple proposition and it will be also related to the topic ??",
    "a",
    "any words can be writed now ?"
]

# test
file_path = 'example.txt'  # Replace 'your_file.txt' with the path to your text file
file_content = ""
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        # print(file_content)
except FileNotFoundError:
    print(f"The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
new_messages = parser.parsing_telegram(file_content)
# only testing

# Tokenize and encode the new messages
new_encodings = tokenizer(new_messages, truncation=True, padding=True, return_tensors='pt')

# Make predictions on the new messages
model.eval()
with torch.no_grad():
    new_outputs = model(**new_encodings)
    new_logits = new_outputs.logits

# Calculate probabilities using softmax
new_probabilities = torch.softmax(new_logits, dim=1)
human_trafficking_probabilities = new_probabilities[:, 1].tolist()

# Display the probabilities for the new messages
for message, probability in zip(new_messages, human_trafficking_probabilities):
    print(f"Message: {message}")
    print(f"Probability of human trafficking: {probability:.4f}\n")
