import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# List of valid two-letter words
valid_two_letter_words = ['is', 'it', 'of', 'on', 'to', 'by', 'he', 'we', 'me', 'be', 'no', 'do']

# Function to generate two-letter subsequences from a word
def generate_two_letter_subsequences(word):
    return [word[i:i+2] for i in range(len(word) - 1)]

# Create a dataset of words and their two-letter subsequences
words = ['computer', 'information', 'education', 'hello', 'world', 'is', 'it', 'machine', 'learning']
data = []
labels = []

# Loop through the words and generate features
for word in words:
    subsequences = generate_two_letter_subsequences(word)
    for seq in subsequences:
        data.append([seq])  # Sequence as feature
        labels.append(1 if seq in valid_two_letter_words else 0)  # Label: 1 for valid, 0 for invalid

# Label encoding the characters
encoder = LabelEncoder()
encoder.fit([chr(i) for i in range(97, 123)])  # Encoding lower case letters 'a' to 'z'

# Convert the two-letter subsequences to a numerical feature vector
X = np.array([encoder.transform(list(seq)).reshape(1, -1) for seq in data]).reshape(len(data), -1)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
svm_model = SVC(kernel='linear')  # Use 'linear' or 'rbf' kernel
svm_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = svm_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
