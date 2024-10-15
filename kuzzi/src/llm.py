# llm_models.py

import os
import boto3
import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
import pydantic
from abc import ABC, abstractmethod
from typing import Optional

from src.vector_store import ChromaVectorStore
from src.embed import AzureOpenAIEmbedder, AWSBedrockEmbedder

# Load environment variables from .env file
load_dotenv()

# Pydantic model for structured responses
class Answer(pydantic.v1.BaseModel):
    """An answer to the user question asked by the user."""
    answer: str

# Base class for the LLM models
class LLMModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Abstract method to generate a response from the LLM model based on the prompt and optional context.
        """
        pass

    def create_system_message(self, context: str = "") -> list:
        system_prompt = (
            "You are an SQL query generator. Please generate SQL queries based on the user's requests.\n"
            "Do not include any comments in the SQL query, return only the query itself."
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
        self.client = init_chat_model(
            "gpt-4o",
            model_provider="azure_openai",
            temperature=0.7,
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_CHAT_API_BASE"),
            api_key=os.getenv("AZURE_CHAT_API_KEY"),
            api_version=os.getenv("AZURE_CHAT_API_VERSION", "2024-08-01-preview")
        )

    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Azure OpenAI's GPT models."""
        try:
            messages = self.create_system_message(context)
            messages.append(HumanMessage(content=prompt))
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error generating response from Azure: {e}")
            return "Error with Azure OpenAI"

# AWS Bedrock Chat Model
class BedrockChat(LLMModel):
    def __init__(self, model_name: Optional[str] = None):
        super().__init__()
        self.client = init_chat_model(
            model_name or os.getenv("AWS_BEDROCK_CHAT_MODEL", "us.anthropic.claude-3-5-sonnet-20240620-v1:0"),
            model_provider="bedrock",
            temperature=0.7
        )

    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using AWS Bedrock's model."""
        try:
            system_message = self.create_system_message(context)[0].content
            full_prompt = {
                "input": f"{system_message}\nUser: {prompt}\nAssistant:"
            }

            response = self.client.invoke(json.dumps(full_prompt))
            return response.content
        except Exception as e:
            print(f"Error generating response from AWS: {e}")
            return "Error with AWS Bedrock"

# Chat function to generate response with Azure or AWS, using ChromaVectorStore for context
def chat_with_llm(llm_model: LLMModel, prompt: str, vector_store: ChromaVectorStore) -> str:
    """Generate chat response using the specified LLM model (Azure or AWS) and optionally use ChromaDB for context."""
    context = ""
    
    context = vector_store.get_context(prompt)
    print(f"Using context from ChromaDB:\n{context}\n")

    response = llm_model.generate_response(prompt, context)
    return response

def create_llm(provider: str, model_name: Optional[str] = None):
    if provider == 'azure':
        return AzureChat()
    elif provider == 'bedrock':
        return BedrockChat(model_name=model_name)
    else:
        raise ValueError("Invalid provider type")