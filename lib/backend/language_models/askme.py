# imports
import warnings

# Suppress TypedStorage deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message="LangChainDeprecationWarning: The function `run` was deprecated.*")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from chromadb.utils import embedding_functions

import json
import chromadb
from chromadb.config import Settings
from chromadb import EmbeddingFunction
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def get_context_with_max_similarity(user_question):
    # Initialize SentenceTransformer model
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L12-v2", normalize_embeddings=True)

    # Embed the user's question
    user_question_embedding = sentence_transformer_ef([user_question])[0]

    # Perform the query using Chroma
    search_results = collection_2013.query(query_embeddings=[user_question_embedding], n_results=5)

    # Find the index of the context with the maximum similarity score
    max_similarity_index = search_results['distances'][0].index(max(search_results['distances'][0]))

    # Get the context with the maximum similarity score
    context_with_max_similarity = search_results['documents'][0][max_similarity_index]

    return context_with_max_similarity

chroma_internet_client = chromadb.HttpClient(host='16.171.68.145', port=8000, settings=Settings(allow_reset=True))

collection_2013 = chroma_internet_client.get_collection('2013pubmed')

# Initialize SentenceTransformer model
sentence_transformer_ef = SentenceTransformer('all-MiniLM-L12-v2')

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


# Initialize the LLMChain

llm_chain_multi_query= LLMChain(prompt=MULTI_QUERY_PROMPT_MODEL, llm=llm)
llm_chain_answer = LLMChain(prompt=QUERY_PROMPT_MODEL, llm=llm)

def generate_questions(context):
    # Generate questions based on the provided context
    generated_questions = llm_chain_multi_query.run({'context': context})
    
    return generated_questions


def main():
    while True:
        user_question = input("Enter your question (or 'exit' to quit): ")
        if user_question.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        context_with_max_similarity = get_context_with_max_similarity(user_question)

        generated_questions = generate_questions(context_with_max_similarity)

        
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
        
        
        print("ANSWER")
        answer = llm_chain_answer.run({'context': new_context, 'question': user_question})
        print(answer)

if __name__ == "__main__":
    main()