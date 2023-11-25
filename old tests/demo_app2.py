import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Specify the path to your files
negative = 'negative.txt'
positive = 'positive.txt'
test_csv_path = 'test.csv'
# Example data
negative_examples = []
positive_examples = []

# Read bad words from the file
with open(positive, 'r') as file:
    bad_words_content = file.read()
    positive_examples = bad_words_content.split('\n')

# Read bad words from the file
with open(negative, 'r') as file:
    bad_words_content = file.read()
    negative_examples = bad_words_content.split('\n')

# Create a DataFrame
data = pd.DataFrame({"text": positive_examples + negative_examples, "label": [1] * len(positive_examples) + [0] * len(negative_examples)})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Create a pipeline with a bag-of-words model and a Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Example usage
user_input = "hello can we write something but if we would talk about the human trafficking? "
probability = model.predict_proba([user_input])[0, 1]
print(f"Probability of human trafficking: {probability}")