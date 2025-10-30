import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path=r"D:\OSPO\KG-RAG1\chroma_store_abstracts")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/e5-base-v2")

col = client.get_collection("abstracts_all", embedding_function=embed_fn)
print("Docs:", col.count())
sample = col.peek(3)
print(sample)
