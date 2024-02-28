# app.py

from flask import Flask, render_template, request
from create_collection import my_collection, sentence_transformer_ef

app = Flask(__name__)

# Function to get search results for a user question
def get_search_results(user_question):
    user_question_embedding = sentence_transformer_ef([user_question])[0]
    search_results = my_collection.query(
        query_embeddings=[user_question_embedding], n_results=5
    )
    return search_results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_question = request.form['query']
        if user_question:
            search_results = get_search_results(user_question)
            return render_template('index.html', query=user_question, results=search_results)

    return render_template('index.html', query=None, results=None)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
