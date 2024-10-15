from langchain_community.vectorstores import Chroma
from abc import ABC, abstractmethod
import lancedb  
import uuid  
from src.embed import Embedder

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

class ChromaVectorStore(VectorStore):
    def __init__(self, embedder, collection_name: str = "default"):
        super().__init__(embedder, collection_name)
        self.persist_directory = "./chroma_storage"
        self.vector_store = None

    def build_vector_store(self, texts: list):
        """Build the Chroma vector store from a list of texts."""
        if not self.vector_store:
            self.vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embedder,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
        else:
            self.vector_store.add_texts(texts)
        print(f"Vector store '{self.collection_name}' updated with {len(texts)} texts.")

    def add_sample(self, text: str):
        """Add a single sample to the Chroma vector store."""
        if not self.vector_store:
            self.vector_store = Chroma.from_texts(
                texts=[text],
                embedding=self.embedder,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
        else:
            self.vector_store.add_texts([text])
        print(f"Added sample to vector store: {text}")
    
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
        search_results = self.vector_store.similarity_search(query, k=n_results)

        # Extract context from search results
        context = "\n".join([doc.page_content for doc in search_results])
        return context

class LanceDBVectorStore(VectorStore):
    def __init__(self, embedder, collection_name: str = "default"):
        super().__init__(embedder, collection_name)
        self.db = lancedb.connect("./lance_storage")  # Connect to the local LanceDB
        self.collection_name = collection_name
        self.collection = None

    def build_vector_store(self, texts: list):
        """Build the LanceDB vector store from a list of texts."""
        embeddings = self.embedder.embed_documents(texts)
        data = [{"id": str(uuid.uuid4()), "text": texts[i], "embedding": embeddings[i]} for i in range(len(texts))]

        # Create or retrieve collection
        if self.collection_name in self.db.table_names():
            self.collection = self.db.open_table(self.collection_name)
        else:
            self.collection = self.db.create_table(self.collection_name, data[0].keys())

        # Insert data into the collection
        self.collection.add(data)
        print(f"Vector store '{self.collection_name}' created/updated with {len(texts)} texts.")

    def add_sample(self, text: str):
        """Add a single sample to the LanceDB vector store."""
        embedding = self.embedder.embed_documents([text])
        data = {"id": str(uuid.uuid4()), "text": text, "embedding": embedding}

        # Create or retrieve collection
        if self.collection_name in self.db.table_names():
            self.collection = self.db.open_table(self.collection_name)
        else:
            self.collection = self.db.create_table(self.collection_name, data.keys())

        # Insert the single data entry into the collection
        self.collection.add([data])
        print(f"Added sample to LanceDB: {text}")

    def search(self, query: str, n_results: int = 5):
        """Search for similar vectors in the LanceDB collection."""
        embedding = self.embedder.embed_documents([query])
        if not self.collection:
            print("Collection not initialized.")
            return []

        # Perform similarity search
        results = self.collection.search(embedding).limit(n_results)
        return results
    
    def get_context(self, query: str, n_results: int = 3) -> str:
        """Retrieve relevant context from LanceDB for a given query."""
        embedding = self.embedder.embed_documents([query])
        if not self.collection:
            print("Collection not initialized.")
            return ""   
        # Perform similarity search
        results = self.collection.search(embedding).limit(n_results)
        return results

def create_vector_store(provider: str, collection_name: str, embedder: Embedder):
    if provider == 'chroma':
        return ChromaVectorStore(embedder, collection_name)
    elif provider == 'lance':
        return LanceDBVectorStore(embedder, collection_name)
    else:
        raise ValueError("Invalid provider type")