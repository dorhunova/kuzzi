from langchain.vectorstores import Chroma
from abc import ABC, abstractmethod
from chromadb import Client
from chromadb.config import Settings

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
