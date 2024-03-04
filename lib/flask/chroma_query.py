import chromadb
from chromadb.utils import embedding_functions



def get_context_with_max_similarity(user_question):
    try:
        # Connect to ChromaDB
        chroma_internet_client = chromadb.HttpClient(host='16.171.68.145', port=8000)

        print(chroma_internet_client.list_collections())

        # Access the 'pubmed_whole' collection
        collection_2013 = chroma_internet_client.get_collection('pubmed_whole')



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
    except Exception as e:
        print("Error:", e)
        return None

# Test the function
user_question = 'CASK Disorder'
context = get_context_with_max_similarity(user_question)
if context:
    print("Context with Maximum Similarity Score:")
    print(context)
else:
    print("Failed to retrieve context.")
