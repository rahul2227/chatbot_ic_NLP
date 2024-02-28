# create_collection.py

import json
from chromadb import Client
from chromadb.utils import embedding_functions

# Specify the path to your JSON file
json_file_path = '/Users/vasu/Desktop/project/chatbot_ic_NLP/lib/NLP-Models/2013pubmed.json'

# Open the JSON file and load the data
with open(json_file_path, 'r') as json_file:
    dataset = json.load(json_file)

# Print the loaded dataset
print(len(dataset))

# Import Chroma and instantiate a client.
# The default Chroma client is ephemeral, meaning it will not save to disk.
client = Client()

# Import embedding functions and SentenceTransformer
from sentence_transformers import SentenceTransformer

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="multi-qa-MiniLM-L6-cos-v1"
)

# Create a collection
my_collection = client.create_collection(
    "2013pubmed", embedding_function=sentence_transformer_ef
)

# Extract data from the dataset and store it in the collection
my_collection.add(
    ids=[str(entry['PMID']) for entry in dataset],
    documents=[entry['Abstract'] for entry in dataset],
    metadatas=[
        {'title': entry['Title'], 'author': entry['Author']} for entry in dataset
    ],
)
print(len(dataset))