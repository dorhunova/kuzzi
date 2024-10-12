import os
from dotenv import load_dotenv
import logging

from src.connectors import PostgresConnector
from src.embed import AzureOpenAIEmbedder, AWSBedrockEmbedder
from src.vector_store import ChromaVectorStore

# Load environment variables from the .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)

def main():
    # Create an instance of PostgresConnector
    pg_conn = PostgresConnector()

    # Connect to the PostgreSQL database using environment variables
    pg_conn.connect()

    # Test query to run (this is just an example, you can adjust it based on your database schema)
    test_query = "select * from tickets limit 10;"

    # Run the query and print the result
    result = pg_conn.run(test_query)
    
    print("Query execution result: ")
    print(result)
    
    
# Example usage:
def test_embedder_azure():
    embedder = AzureOpenAIEmbedder()
    embedding = embedder.embed_query("Hello world")
    print(embedding)

def test_embedder_aws():
    embedder = AWSBedrockEmbedder()
    embedding = embedder.embed_query("Hello world")
    print(embedding)
    
# Example usage with Azure OpenAI and AWS Bedrock embedders

def test_chroma_with_azure():
    texts = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome."
    ]
    query = "What is the capital of France?"

    # Use Azure OpenAI Embedder
    azure_embedder = AzureOpenAIEmbedder()
    vector_store = ChromaVectorStore(azure_embedder, collection_name="azure_embeddings")

    # Build the vector store
    vector_store.build_vector_store(texts)

    # Perform search
    results = vector_store.search(query)
    print("Search Results with Azure:", results)


def test_chroma_with_bedrock():
    texts = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome."
    ]
    query = "What is the capital of Germany?"

    # Use AWS Bedrock Embedder
    bedrock_embedder = AWSBedrockEmbedder()
    vector_store = ChromaVectorStore(bedrock_embedder, collection_name="bedrock_embeddings")

    # Build the vector store
    vector_store.build_vector_store(texts)

    # Perform search
    results = vector_store.search(query)
    print("Search Results with Bedrock:", results)


if __name__ == "__main__":
    main()
    # test_embedder_azure()
    # test_embedder_aws()
    test_chroma_with_azure()  # Test with Azure OpenAI Embedder
    test_chroma_with_bedrock()  # Test with AWS Bedrock Embedder
