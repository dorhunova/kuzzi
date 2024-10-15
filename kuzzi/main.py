import os
from dotenv import load_dotenv
import logging

from fastapi import FastAPI, HTTPException

from src.connectors import PostgresConnector
from src.embed import AzureOpenAIEmbedder, AWSBedrockEmbedder, create_embedder
from src.vector_store import ChromaVectorStore, LanceDBVectorStore, create_vector_store
from src.llm import chat_with_llm, AzureChat, BedrockChat, create_llm
from src.train.trainer import Trainer
from src.data_models import ChatRequest, ChatResponse
# Load environment variables from the .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI()


logging.info("Starting up the FastAPI application, setting up Postgres connection, creating vector store...")
pg_conn = PostgresConnector()
pg_conn.connect()

provider = os.getenv("PROVIDER")
vector_store_type = os.getenv("VECTOR_STORE")

embedder = create_embedder(provider)
vector_store = create_vector_store(vector_store_type, collection_name=f"{provider}-embeddings", embedder=embedder)
trainer = Trainer(vector_store)
trainer.load_from_yaml(os.getenv("TRAINING_DATA_PATH"))
chat_model = create_llm(provider)
    

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = chat_with_llm(chat_model, request.question, vector_store)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/summary")
async def summary_endpoint():
    # Placeholder for summary functionality
    return {"message": "Summary endpoint not implemented yet"}

@app.get("/plot")
async def plot_endpoint():
    # Placeholder for plot functionality
    return {"message": "Plot endpoint not implemented yet"}
    
    
    
    
