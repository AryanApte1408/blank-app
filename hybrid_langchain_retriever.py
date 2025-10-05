#hybrid_langchain retriever.py
import hashlib
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import config_full as config
from graph_retriever import get_papers_by_researcher, get_paper_neighbors

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

def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8):
    # Step 1: Chroma
    chroma_docs = chroma.similarity_search(query, k=k_chroma)

    # Step 2: Extract candidates
    titles = {d.metadata.get("title") for d in chroma_docs if d.metadata.get("title")}
    researchers = {d.metadata.get("researcher") for d in chroma_docs if d.metadata.get("researcher")}

    # Step 3: Query Neo4j for enrichment
    graph_hits = []
    for r in researchers:
        graph_hits.extend(get_papers_by_researcher(r, k=k_graph))
    for t in titles:
        graph_hits.extend(get_paper_neighbors(t, k_authors=5))

    # Step 4: Fuse
    fused = []
    for d in chroma_docs:
        fused.append(f"[Chroma] {d.metadata.get('title','Untitled')} ({d.metadata.get('publication_date','?')})\n{d.page_content[:300]}")
    for g in graph_hits:
        fused.append(f"[Graph] {g.get('title','Untitled')} ({g.get('year','?')}) — {', '.join(g.get('authors', []))}")

    return {
        "graph_hits": graph_hits,
        "chroma_ctx": [d.page_content for d in chroma_docs],
        "fused_text_blocks": _dedupe(fused),
    }
