from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Sample training data
emails = ["Hello, how are you?", "Win a free iPhone now!", "Meeting at 3 PM.", "Claim your prize money!"]
labels = ["Not Spam", "Not Spam", "Not Spam", "Spam"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.25, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create a Multinomial Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()

# Train the classifier
naive_bayes_classifier.fit(X_train_vectorized, y_train)

# Predict probabilities on the test set
probabilities = naive_bayes_classifier.predict_proba(X_test_vectorized)

# Display class probabilities for each instance in the test set
for i, email in enumerate(X_test):
    print(f"\nEmail: {email}")
    print("Class Probabilities:")
    for j, class_label in enumerate(naive_bayes_classifier.classes_):
        print(f"{class_label}: {probabilities[i][j]}")

# Predict on the test set
y_pred = naive_bayes_classifier.predict(X_test_vectorized)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)

# Save the trained model
with open('naive_bayes_model.pkl', 'wb') as model_file:
    pickle.dump(naive_bayes_classifier, model_file)

# Save the vectorizer
with open('count_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

import os    
os.getcwd()
