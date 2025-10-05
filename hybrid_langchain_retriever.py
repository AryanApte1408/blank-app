import hashlib
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import config_full as config
from graph_retriever import search_graph

device = "cuda" if torch.cuda.is_available() else "cpu"
embed = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",
    model_kwargs={"device": device}
)

client = chromadb.PersistentClient(path=config.CHROMA_DIR)
chroma = Chroma(
    client=client,
    collection_name="papers_all",
    embedding_function=embed
)

def _dedupe(lst):
    seen, out = set(), []
    for x in lst:
        h = hashlib.sha1(x.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(x)
    return out

def hybrid_search(query: str, k_graph: int = 8, k_chroma: int = 8):
    graph_hits = search_graph(query, k=k_graph)
    chroma_docs = chroma.similarity_search(query, k=k_chroma)

    fused = []
    for h in graph_hits:
        fused.append(f"[Graph] {h.get('title')} ({h.get('year')})")
    for d in chroma_docs:
        fused.append(f"[Chroma] {d.page_content[:200]}")

    return {
        "graph_hits": graph_hits,
        "chroma_ctx": [d.page_content for d in chroma_docs],
        "fused_text_blocks": _dedupe(fused),
    }
