import os
from dotenv import load_dotenv
import logging

from src.connectors import PostgresConnector
from src.embed import AzureOpenAIEmbedder, AWSBedrockEmbedder
from src.vector_store import ChromaVectorStore, LanceDBVectorStore
from src.llm import chat_with_llm, AzureChat, BedrockChat
from src.train.trainer import Trainer
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
def test_vector_store_with_embedder(llm_type: str, vector_store_type: str):
    texts = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome."
    ]
    query = "What is the capital of France?"

    # Initialize the embedder based on the llm_type
    if llm_type == 'azure':
        embedder = AzureOpenAIEmbedder()
    elif llm_type == 'bedrock':
        embedder = AWSBedrockEmbedder()
    else:
        raise ValueError("Invalid LLM type")

    # Initialize the vector store based on the vector_store_type
    if vector_store_type == 'chroma':
        vector_store = ChromaVectorStore(embedder, collection_name=f"{llm_type}_embeddings")
    elif vector_store_type == 'lance':
        vector_store = LanceDBVectorStore(embedder, collection_name=f"{llm_type}_embeddings")
    else:
        raise ValueError("Invalid vector store type")

    # Build the vector store
    vector_store.build_vector_store(texts)

    # Perform search
    results = vector_store.search(query)
    print(f"Search Results with {llm_type.capitalize()} and {vector_store_type.capitalize()}: {results}")
    return results

def test_llm_with_vector_store_and_embedder(llm_type: str = 'azure', vector_store_type: str = 'chroma'):
    # Initialize the embedder based on the llm_type
    if llm_type == 'azure':
        embedder = AzureOpenAIEmbedder()
    elif llm_type == 'bedrock':
        embedder = AWSBedrockEmbedder()
    else:
        raise ValueError("Invalid LLM type")
    
    # Initialize the vector store based on the vector_store type
    if vector_store_type == 'chroma':
        vector_store = ChromaVectorStore(embedder, collection_name=f"{llm_type}-embeddings")
    elif vector_store_type == 'lance':
        vector_store = LanceDBVectorStore(embedder, collection_name=f"{llm_type}-embeddings")  
    else:
        raise ValueError("Invalid vector store type")
    
    # Initialize the chat model based on the llm_type
    if llm_type == 'azure':
        chat_model = AzureChat()
    elif llm_type == 'bedrock':
        chat_model = BedrockChat()
    else:
        raise ValueError("Invalid LLM type")
    
    prompt = "What is the capital of Germany?"

    texts = ["The capital of France is Paris.", "The capital of Germany is Berlin.", "The capital of Italy is Rome."]
    vector_store.build_vector_store(texts)

    response = chat_with_llm(chat_model, prompt, vector_store, use_chroma=True)
    print(f"{llm_type.capitalize()} Response:\n{response}\n")
    
def test_trainer_with_vector_store_and_embedder(vector_store_type: str, llm_type: str):
    # Initialize the embedder based on the llm_type
    if llm_type == 'azure':
        embedder = AzureOpenAIEmbedder()
    elif llm_type == 'bedrock':
        embedder = AWSBedrockEmbedder()
    else:
        raise ValueError("Invalid LLM type")
    
    if vector_store_type == 'chroma':
        vector_store = ChromaVectorStore(embedder, collection_name=f"{llm_type}-embeddings")
    elif vector_store_type == 'lance':
        vector_store = LanceDBVectorStore(embedder, collection_name=f"{llm_type}-embeddings")
    else:
        raise ValueError("Invalid vector store type")
    
    trainer = Trainer(vector_store=vector_store)
    trainer.load_from_yaml(yaml_path=os.getenv("TRAINING_DATA_PATH"))
    
    chat_model = AzureChat()
    
    prompt = "How to find all of the request types from the it department?" 
    
    response = chat_with_llm(chat_model, prompt, vector_store)
    print(f"Azure Response:\n{response}\n")
    import ipdb; ipdb.set_trace()
    
    
    

if __name__ == "__main__": 
    debug = False 
    if debug:
        logging.info("Starting Kuzzi")
        logging.info("Connecting to Postgres, testing query execution...")
        main()
        
        logging.info("Testing Embedder with Azure and Bedrock")
        test_embedder_azure()
        test_embedder_aws()
        
        logging.info("Testing Chroma with Azure and Bedrock")
        test_vector_store_with_embedder(llm_type='azure', vector_store_type='chroma')
        test_vector_store_with_embedder(llm_type='bedrock', vector_store_type='chroma')
        # test_vector_store_with_embedder(llm_type='azure', vector_store_type='lance')
        # test_vector_store_with_embedder(llm_type='bedrock', vector_store_type='lance')
        
        logging.info("Testing LLM with vector store")
        
        logging.info("Testing LLM with Azure and Chroma")
        test_llm_with_vector_store_and_embedder(llm_type='azure', vector_store_type='chroma')
        logging.info("Testing LLM with Bedrock and Chroma")
        test_llm_with_vector_store_and_embedder(llm_type='bedrock', vector_store_type='chroma')
        # logging.info("Testing LLM with Azure and LanceDB")
        # test_llm_with_vector_store_and_embedder(llm_type='azure', vector_store_type='lance')
        # logging.info("Testing LLM with Bedrock and LanceDB")
        # test_llm_with_vector_store_and_embedder(llm_type='bedrock', vector_store_type='lance')

    logging.info("Testing Trainer with vector store and embedder")
    test_trainer_with_vector_store_and_embedder(vector_store_type='chroma', llm_type='azure')
    test_trainer_with_vector_store_and_embedder(vector_store_type='chroma', llm_type='bedrock')
    # test_trainer_with_vector_store_and_embedder(vector_store_type='lance', llm_type='azure')
    
        