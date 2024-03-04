from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import chromadb
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




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/answer', methods=['POST'])



def answer():
    if request.method == 'POST':
        # Get the user's question from the form
        user_question = request.form['question']
        # Turbo setup
        OPEN_API_KEY = "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"
        llm = OpenAI(api_key=OPEN_API_KEY)

        QUERY_PROMPT_MODEL = PromptTemplate(
            input_variables=["context"],
            template="""
        You are an AI language model assistant. Your task is to generate answer
            by taking information from the relevant context provided from a vector 
            database. By considering multiple perspectives on the user question, your goal is to help
            the user understand the concept of the question asked that is also relevant to the context provided. 
            Provide these answers with proper type setting.

            Original question: {question}
            Context : {context}
        """,)

        MULTI_QUERY_PROMPT_MODEL = PromptTemplate(
            input_variables=["context"],
            template="""You are an AI language model assistant. Your task is to generate questions based on the provided context. 
            Generate list of 3 questions that help to explore different aspects of the context and deepen the understanding of the topic. 
            Provide these questions with proper type setting.

            
            Context : {context}
        """,)



        llm_chain_multi_query= LLMChain(prompt=MULTI_QUERY_PROMPT_MODEL, llm=llm)
        llm_chain_answer = LLMChain(prompt=QUERY_PROMPT_MODEL, llm=llm)
        
        # Get the context with maximum similarity using chroma_query function
        context_with_max_similarity = get_context_with_max_similarity(user_question)

        
        generated_questions = llm_chain_multi_query.run({'context': context_with_max_similarity})

        #FETCHING 3 questions

        # Split the generated questions string into a list of individual questions
        generated_questions_list = generated_questions.split('\n')

        # Filter out any empty strings from the list
        generated_questions_list = [question.strip() for question in generated_questions_list if question.strip()]

        # Extract the first three questions from the list
        question1 = generated_questions_list[0] if len(generated_questions_list) > 0 else ""
        question2 = generated_questions_list[1] if len(generated_questions_list) > 1 else ""
        question3 = generated_questions_list[2] if len(generated_questions_list) > 2 else ""



        q1_with_max_simialrity=get_context_with_max_similarity(question1)
        q2_with_max_simialrity=get_context_with_max_similarity(question2)
        q3_with_max_simialrity=get_context_with_max_similarity(question3)



        new_context=context_with_max_similarity+q1_with_max_simialrity+q2_with_max_simialrity+q3_with_max_simialrity
        # print("ANSWER")
        answer = llm_chain_answer.run({'question': user_question,'context': new_context, })
        
        
        return render_template('answer.html', answer=answer)




if __name__ == '__main__':
    app.run(debug=True)
