from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

app = Flask(__name__)

# Load the trained model
with open('naive_bayes_model.pkl', 'rb') as model_file:
    naive_bayes_classifier = pickle.load(model_file)

# Load the vectorizer
with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = [request.form['email']]
        email_vectorized = vectorizer.transform(email)
        prediction = naive_bayes_classifier.predict(email_vectorized)
        probability = naive_bayes_classifier.predict_proba(email_vectorized)

        return render_template('result.html', email=email[0], prediction=prediction[0], probability=probability[0])

if __name__ == '__main__':
    app.run(debug=True)
