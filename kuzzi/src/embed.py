from langchain_openai import AzureOpenAIEmbeddings
from langchain_aws import BedrockEmbeddings
import boto3
import os
from dotenv import load_dotenv
import logging
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Base class for all embedders
class Embedder:
    def embed_query(self, text: str):
        result = self.embedder.embed_query(text)
        return result

    def embed_documents(self, texts: list):
        result = self.embedder.embed_documents(texts)
        return result


# Embedder for Azure OpenAI using langchain
class AzureOpenAIEmbedder(Embedder):
    def __init__(self, deployment_name: str = None, chunk_size: int = 1024):
        """
        Initialize Azure OpenAI Embedder with deployment and API version.
        """
        self.embedder = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
            azure_endpoint=os.getenv("AZURE_EMBEDDINGS_API_BASE"),
            api_key=os.getenv("AZURE_EMBEDDINGS_API_KEY"),
            openai_api_version=os.getenv("AZURE_EMBEDDINGS_API_VERSION"),
            chunk_size=chunk_size
        )

# Embedder for AWS Bedrock
class AWSBedrockEmbedder(Embedder):
    def __init__(self, model_name: str = None):
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv("AWS_DEFAULT_REGION"), 
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN")
        )
        self.embedder = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id=model_name or os.getenv("AWS_BEDROCK_EMBEDDINGS_MODEL", "cohere.embed-english-v3")
        )


def create_embedder(provider: str, model_name: Optional[str] = None):
    if provider == 'azure':
        return AzureOpenAIEmbedder()
    elif provider == 'bedrock':
        return AWSBedrockEmbedder(model_name)
    else:
        raise ValueError("Invalid provider type")