import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Specify the path to your files
bad_words_file_path = 'bad_words.txt'
test_csv_path = 'test.csv'

# Read bad words from the file
with open(bad_words_file_path, 'r') as file:
    bad_words_content = file.read()
    bad_words = bad_words_content.split('\n')

# Read test data from the CSV file
test_data = pd.read_csv(test_csv_path)

# Extract messages and labels from the test data
messages = test_data['Messages'].tolist()
labels = test_data['Labels'].tolist()

# Create a pipeline with a bag-of-words model and a Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Now you can use the trained model to predict the probability of a message being potentially vulnerable
def predict_vulnerability(message):
    probability = model.predict_proba([message])[0, 1]
    return probability

# Example usage
# user_input = input("Enter a message: ")
user_input = "a"
probability = predict_vulnerability(user_input)
print(f"Probability of vulnerability: {probability}")
