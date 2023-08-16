import os
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from langchain.embeddings import OpenAIEmbeddings
from chromadb.api.types import Documents, Embeddings

class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, *args, **kwargs):
        self.embeddings = OpenAIEmbeddings(*args, openai_api_key=os.environ["OPENAI_API_KEY"], **kwargs)

        # do some initialization
    def __call__(self, texts: Documents) -> Embeddings:
        return self.embeddings.embed_documents(texts)

ada_embedding_function = OpenAIEmbeddingFunction()

