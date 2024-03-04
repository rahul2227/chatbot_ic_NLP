from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import json
from chroma_query import get_context_with_max_similarity

app = Flask(__name__)




index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ask a Question</title>
</head>
<body>
    <h1>Ask a Question</h1>
    <form action="/answer" method="POST">
        <label for="question">Enter your question:</label><br>
        <input type="text" id="question" name="question" required><br>
        <button type="submit">Submit</button>
    </form>
</body>
</html>
"""

answer_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Answer</title>
</head>
<body>
    <h1>Answer</h1>
    <p>{{ answer }}</p>
</body>
</html>
"""


# Initialize SentenceTransformer model
sentence_transformer_ef = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Chroma DB setup
chroma_internet_client = chromadb.HttpClient(host='16.171.68.145', port=8000)

# Turbo setup
OPEN_API_KEY = "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"
llm = OpenAI(api_key=OPEN_API_KEY)
QUERY_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template = """You are an AI language model assistant. Your task is to generate answer
    by taking information from the relevant context provided from a vector 
    database. By considering multiple perspectives on the user question, your goal is to help
    the user understand the concept of the question asked that is also relevant to the context provided. 
    Provide these answers with proper type setting.

    Original question: {question}
    Context : {context}
    """,
)

# prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=QUERY_PROMPT, llm=llm)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    if request.method == 'POST':
        # Get the user's question from the form
        user_question = request.form['question']
        
        # Get the context with maximum similarity using chroma_query function
        context_with_max_similarity = get_context_with_max_similarity(user_question)
        
        # Generate an answer using Turbo based on the context with maximum similarity

        new_context= generate_questions(context_with_max_similarity)
        new=new_context+ context_with_max_similarity
        answer = generate_answer(user_question,new)
        
        return render_template('answer.html', answer=answer)

def generate_answer(question, context):
    # Run Turbo to generate an answer
    answer = llm_chain.run({'context': context, 'question': question})
    
    return answer

def generate_questions(context):
    # Generate questions based on the provided context
    generated_questions = llm_chain.run({'context': context})
    
    return generated_questions

if __name__ == '__main__':
    app.run(debug=True)
