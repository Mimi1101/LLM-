import json
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

chroma_client = chromadb.PersistentClient(path="my_chromadb")
load_dotenv('env')

# Use ada-002 for embedding
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

collection = chroma_client.get_collection(
    name="documents",
    embedding_function=openai_ef
)