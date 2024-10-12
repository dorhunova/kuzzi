# llm_models.py

import os
import boto3
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Import the updated ChromaVectorStore
from chromadb import Client
from chromadb.config import Settings
from abc import ABC, abstractmethod
from langchain.vectorstores import Chroma

from src.embed import OpenAIEmbeddings, AzureEmbeddings

# Load environment variables from .env file
load_dotenv()

# Base class for Vector Store
class VectorStore(ABC):
    def __init__(self, embedder, collection_name: str):
        self.embedder = embedder
        self.collection_name = collection_name
        self.vector_store = None

    @abstractmethod
    def build_vector_store(self, texts: list):
        """Abstract method to build a vector store from a list of texts."""
        pass

    @abstractmethod
    def search(self, query: str, n_results: int):
        """Abstract method to search for vectors in the vector store."""
        pass

# Chroma Vector Store implementation
class ChromaVectorStore(VectorStore):
    def __init__(self, embedder, collection_name: str = "default"):
        super().__init__(embedder, collection_name)
        self.persist_directory = "./chroma_storage"

    def build_vector_store(self, texts: list):
        """Build the Chroma vector store from a list of texts."""
        self.vector_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embedder,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        print(f"Vector store '{self.collection_name}' created with {len(texts)} texts.")

    def search(self, query: str, n_results: int = 5):
        """Search for similar vectors in the vector store."""
        if not self.vector_store:
            # Load the existing vector store if not already loaded
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedder,
                persist_directory=self.persist_directory
            )
        results = self.vector_store.similarity_search(query, k=n_results)
        return results

    def get_context(self, query: str, n_results: int = 3) -> str:
        """Retrieve relevant context from ChromaDB for a given query."""
        if not self.vector_store:
            # Load the existing vector store if not already loaded
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedder,
                persist_directory=self.persist_directory
            )

        # Perform similarity search
        search_results = self.vector_store.similarity_search(query, k=n_results)

        # Extract context from search results
        context = "\n".join([doc.page_content for doc in search_results])
        return context

# Base class for the LLM models
class LLMModel:
    def __init__(self):
        pass

    def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Generate a response from the LLM model based on the prompt and optional context.
        """
        pass

    def create_system_message(self, context: str = "") -> list:
        system_prompt = (
            "You are an SQL query generator. Please generate SQL queries based on the user's requests.\n"
        )
        if context:
            examples_prompt = (
                "Here are some relevant database schema and examples:\n"
                f"{context}\n"
                "Make sure to use the correct syntax for the SQL query.\n"
            )
            return [SystemMessage(content=system_prompt + examples_prompt)]
        else:
            return [SystemMessage(content=system_prompt)]

# Azure OpenAI Chat Model
class AzureChat(LLMModel):
    def __init__(self):
        super().__init__()
        self.api_base = os.getenv("AZURE_CHAT_API_BASE")
        self.api_version = os.getenv("AZURE_CHAT_API_VERSION", "2023-05-15")
        self.api_key = os.getenv("AZURE_CHAT_API_KEY")
        self.deployment = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4")

        # Set up the OpenAI API client for Azure
        self.client = ChatOpenAI(
            openai_api_base=self.api_base,
            openai_api_version=self.api_version,
            openai_api_key=self.api_key,
            model_name=self.deployment,
            temperature=0.7,
            max_tokens=1000,
            openai_api_type="azure"
        )

    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Azure OpenAI's GPT models."""
        try:
            messages = self.create_system_message(context)
            messages.append(HumanMessage(content=prompt))

            response = self.client(messages)
            return response.content
        except Exception as e:
            print(f"Error generating response from Azure: {e}")
            return "Error with Azure OpenAI"

# AWS Bedrock Chat Model
class BedrockChat(LLMModel):
    def __init__(self):
        super().__init__()
        self.region_name = os.getenv("AWS_DEFAULT_REGION")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = os.getenv("AWS_SESSION_TOKEN")
        self.model_name = os.getenv("AWS_BEDROCK_MODEL", "anthropic.claude-2")

        # Set up the Bedrock client
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token
        )

    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using AWS Bedrock's model."""
        try:
            system_message = self.create_system_message(context)[0].content
            full_prompt = f"{system_message}\nUser: {prompt}\nAssistant:"

            response = self.bedrock_client.invoke_model(
                modelId=self.model_name,
                contentType="text/plain",
                accept="text/plain",
                body=full_prompt.encode('utf-8')
            )
            response_body = response['body'].read().decode('utf-8')
            return response_body
        except Exception as e:
            print(f"Error generating response from AWS: {e}")
            return "Error with AWS Bedrock"

# Chat function to generate response with Azure or AWS, using ChromaVectorStore for context
def chat_with_llm(llm_model: LLMModel, prompt: str, vector_store: ChromaVectorStore, use_chroma: bool = False) -> str:
    """Generate chat response using the specified LLM model (Azure or AWS) and optionally use ChromaDB for context."""
    context = ""
    if use_chroma:
        context = vector_store.get_context(prompt)
        print(f"Using context from ChromaDB:\n{context}\n")

    response = llm_model.generate_response(prompt, context)
    return response

# Example usage
if __name__ == "__main__":
    # Initialize the embedder and vector store
    embedder = OpenAIEmbeddings()
    vector_store = ChromaVectorStore(embedder, collection_name="test-collection")

    # Optionally build the vector store if not already built
    texts = 
    vector_store.build_vector_store(texts)

    prompt = "Write an SQL query to find the top 10 customers by sales."

    # Use Azure Chat Model
    azure_model = AzureChat()
    azure_response = chat_with_llm(azure_model, prompt, vector_store, use_chroma=True)
    print(f"Azure Response:\n{azure_response}\n")

    # Use AWS Bedrock Chat Model
    bedrock_model = BedrockChat()
    bedrock_response = chat_with_llm(bedrock_model, prompt, vector_store, use_chroma=True)
    print(f"Bedrock Response:\n{bedrock_response}\n")
