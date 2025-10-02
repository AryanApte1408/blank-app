# # # # # # # chroma_retriever.py
# # # # # # import chromadb
# # # # # # from langchain_chroma import Chroma
# # # # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # # # import config_full as config

# # # # # # client = chromadb.PersistentClient(path=config.CHROMA_DIR)
# # # # # # embedder = HuggingFaceEmbeddings(model_name=config.SENTENCE_TFORMER)

# # # # # # # collection with paper summaries/fulltext already ingested
# # # # # # chroma_meta = Chroma(
# # # # # #     client=client,
# # # # # #     collection_name="paper_metadata",
# # # # # #     embedding_function=embedder
# # # # # # )

# # # # # # def query_chroma(question: str, k: int = 5):
# # # # # #     docs = chroma_meta.similarity_search(question, k=k)
# # # # # #     return [d.page_content for d in docs]

# # # # # # chroma_retriever.py
# # # # # import chromadb
# # # # # from langchain_chroma import Chroma
# # # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # # import config_full as config

# # # # # # Init Chroma client + embedder
# # # # # client = chromadb.PersistentClient(path=config.CHROMA_DIR)
# # # # # embedder = HuggingFaceEmbeddings(model_name=config.SENTENCE_TFORMER)

# # # # # # Register all relevant collections
# # # # # collections = {
# # # # #     "metadata": Chroma(
# # # # #         client=client,
# # # # #         collection_name="paper_metadata",
# # # # #         embedding_function=embedder
# # # # #     ),
# # # # #     "summaries": Chroma(
# # # # #         client=client,
# # # # #         collection_name="paper_summaries",
# # # # #         embedding_function=embedder
# # # # #     ),
# # # # #     "fulltext": Chroma(
# # # # #         client=client,
# # # # #         collection_name="paper_fulltext",
# # # # #         embedding_function=embedder
# # # # #     )
# # # # # }

# # # # # def query_chroma(question: str, k: int = 5):
# # # # #     """
# # # # #     Query all Chroma collections (metadata, summaries, fulltext).
# # # # #     Returns a list of retrieved texts with collection tags.
# # # # #     """
# # # # #     results = []
# # # # #     for name, coll in collections.items():
# # # # #         try:
# # # # #             docs = coll.similarity_search(question, k=k)
# # # # #             for d in docs:
# # # # #                 results.append(f"[{name}] {d.page_content}")
# # # # #         except Exception as e:
# # # # #             print(f"⚠️ Chroma error on {name}: {e}")
# # # # #     return results


# # # # # chroma_retriever.py
# # # # import re
# # # # import chromadb
# # # # from langchain_chroma import Chroma
# # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # import config_full as config

# # # # # Init Chroma client
# # # # client = chromadb.PersistentClient(path=config.CHROMA_DIR)

# # # # # GPU-enabled embeddings (MiniLM-L6-v2 or whatever is in config)
# # # # embedder = HuggingFaceEmbeddings(
# # # #     model_name=config.SENTENCE_TFORMER,
# # # #     model_kwargs={"device": "cuda", "torch_dtype": "float16"}
# # # # )

# # # # # Register all collections
# # # # collections = {
# # # #     "metadata": Chroma(
# # # #         client=client,
# # # #         collection_name="paper_metadata",
# # # #         embedding_function=embedder
# # # #     ),
# # # #     "summaries": Chroma(
# # # #         client=client,
# # # #         collection_name="paper_summaries",
# # # #         embedding_function=embedder
# # # #     ),
# # # #     "fulltext": Chroma(
# # # #         client=client,
# # # #         collection_name="paper_fulltext",
# # # #         embedding_function=embedder
# # # #     )
# # # # }

# # # # # ---------- Context Cleaner ----------
# # # # def clean_context(text: str) -> str:
# # # #     """Remove license junk, DOI-only strings, and giant citation dumps."""
# # # #     t = text.strip()

# # # #     # Kill license/copyright blocks
# # # #     if "license" in t.lower() or "copyright" in t.lower():
# # # #         return ""

# # # #     # Kill DOI-only or URL-only strings
# # # #     if re.match(r"^(https?://|doi)", t, re.I):
# # # #         return ""

# # # #     # Kill long author dumps (>15 capitalized words)
# # # #     if len(re.findall(r"\b[A-Z][a-z]+", t)) > 15:
# # # #         return ""

# # # #     # Collapse whitespace
# # # #     return re.sub(r"\s+", " ", t)

# # # # # ---------- Main Query ----------
# # # # def query_chroma(question: str, k: int = 5):
# # # #     """
# # # #     Query all Chroma collections (metadata, summaries, fulltext).
# # # #     Cleans garbage and tags snippets with their collection.
# # # #     """
# # # #     results = []
# # # #     for name, coll in collections.items():
# # # #         try:
# # # #             docs = coll.similarity_search(question, k=k)
# # # #             for d in docs:
# # # #                 cleaned = clean_context(d.page_content)
# # # #                 if cleaned:
# # # #                     results.append(f"[{name}] {cleaned}")
# # # #         except Exception as e:
# # # #             print(f"⚠️ Chroma error on {name}: {e}")
# # # #     return results

# # # # chroma_retriever.py
# # # import re
# # # import chromadb
# # # from langchain_chroma import Chroma
# # # from langchain_huggingface import HuggingFaceEmbeddings
# # # import config_full as config

# # # # Init Chroma client
# # # client = chromadb.PersistentClient(path=config.CHROMA_DIR)

# # # # GPU-enabled embeddings (force CUDA, no torch_dtype to avoid crash)
# # # embedder = HuggingFaceEmbeddings(
# # #     model_name=config.SENTENCE_TFORMER,
# # #     model_kwargs={"device": "cuda"},       # run on GPU
# # #     encode_kwargs={"normalize_embeddings": True}
# # # )

# # # # Register all collections
# # # collections = {
# # #     "metadata": Chroma(
# # #         client=client,
# # #         collection_name="paper_metadata",
# # #         embedding_function=embedder
# # #     ),
# # #     "summaries": Chroma(
# # #         client=client,
# # #         collection_name="paper_summaries",
# # #         embedding_function=embedder
# # #     ),
# # #     "fulltext": Chroma(
# # #         client=client,
# # #         collection_name="paper_fulltext",
# # #         embedding_function=embedder
# # #     )
# # # }

# # # # ---------- Context Cleaner ----------
# # # def clean_context(text: str) -> str:
# # #     """Remove license junk, DOI-only strings, and giant citation dumps."""
# # #     t = text.strip()

# # #     # Kill license/copyright blocks
# # #     if "license" in t.lower() or "copyright" in t.lower():
# # #         return ""

# # #     # Kill DOI-only or URL-only strings
# # #     if re.match(r"^(https?://|doi)", t, re.I):
# # #         return ""

# # #     # Kill long author dumps (>15 capitalized words)
# # #     if len(re.findall(r"\b[A-Z][a-z]+", t)) > 15:
# # #         return ""

# # #     # Collapse whitespace
# # #     return re.sub(r"\s+", " ", t)

# # # # ---------- Main Query ----------
# # # def query_chroma(question: str, k: int = 5):
# # #     """
# # #     Query all Chroma collections (metadata, summaries, fulltext).
# # #     Cleans garbage and tags snippets with their collection.
# # #     """
# # #     results = []
# # #     for name, coll in collections.items():
# # #         try:
# # #             docs = coll.similarity_search(question, k=k)
# # #             for d in docs:
# # #                 cleaned = clean_context(d.page_content)
# # #                 if cleaned:
# # #                     results.append(f"[{name}] {cleaned}")
# # #         except Exception as e:
# # #             print(f"⚠️ Chroma error on {name}: {e}")
# # #     return results

# # # chroma_retriever.py
# # """
# # Thin Chroma retriever (no hardcoded paths). Uses config_full.py.

# # - Connects to an existing Chroma store with three collections:
# #     * paper_metadata
# #     * paper_summaries
# #     * paper_fulltext
# # - Provides simple similarity search over metadata (titles) by default,
# #   with optional search over summaries and/or fulltext.
# # - Returns both the text and useful metadata for downstream use.

# # Env/Config:
# # - CHROMA_DIR, SENTENCE_TFORMER from config_full.py (env-overridable).
# # """

# # from typing import Dict, List, Literal, Optional, Tuple

# # import chromadb
# # from chromadb.utils import embedding_functions
# # import config_full as config

# # CollectionName = Literal["paper_metadata", "paper_summaries", "paper_fulltext"]

# # # --------------------------- client + collections -----------------------------

# # _client = chromadb.PersistentClient(path=config.CHROMA_DIR)

# # _embed = embedding_functions.SentenceTransformerEmbeddingFunction(
# #     model_name=config.SENTENCE_TFORMER
# # )

# # # Lazily created/attached (no schema assumptions beyond collection names)
# # _coll_meta = _client.get_or_create_collection("paper_metadata", embedding_function=_embed)
# # _coll_sum  = _client.get_or_create_collection("paper_summaries", embedding_function=_embed)
# # _coll_full = _client.get_or_create_collection("paper_fulltext",  embedding_function=_embed)

# # # --------------------------- helpers -----------------------------------------

# # def _do_query(coll, query: str, k: int) -> List[Dict]:
# #     """Wrap Chroma .query() and normalize output."""
# #     if not query or not query.strip():
# #         return []
# #     res = coll.query(query_texts=[query], n_results=max(1, k))
# #     out: List[Dict] = []
# #     # result shape: {"ids":[[...]], "documents":[[...]], "metadatas":[[...]], "distances":[[...]] } (implementation-dependent)
# #     ids        = (res.get("ids") or [[]])[0]
# #     docs       = (res.get("documents") or [[]])[0]
# #     metas_list = (res.get("metadatas") or [[]])[0] or [{} for _ in docs]
# #     dists      = (res.get("distances") or [[]])[0] or [None for _ in docs]

# #     for _id, _doc, _meta, _dist in zip(ids, docs, metas_list, dists):
# #         out.append({
# #             "id": _id,
# #             "text": _doc,
# #             "meta": _meta or {},
# #             "distance": _dist
# #         })
# #     return out

# # def _pick_collections(scope: Optional[List[CollectionName]]) -> List[Tuple[str, any]]:
# #     if not scope:
# #         scope = ["paper_metadata"]
# #     mapping = {
# #         "paper_metadata": ("paper_metadata", _coll_meta),
# #         "paper_summaries": ("paper_summaries", _coll_sum),
# #         "paper_fulltext": ("paper_fulltext", _coll_full),
# #     }
# #     picked: List[Tuple[str, any]] = []
# #     for name in scope:
# #         tup = mapping.get(name)
# #         if tup:
# #             picked.append(tup)
# #     return picked

# # # --------------------------- public API --------------------------------------

# # def query_chroma(
# #     question: str,
# #     k: int = 5,
# #     scope: Optional[List[CollectionName]] = None,
# # ) -> Dict[str, List[Dict]]:
# #     """
# #     Run semantic search against one or more Chroma collections.

# #     Args:
# #         question: user query text
# #         k: results per collection
# #         scope: which collections to search; default ["paper_metadata"].
# #                Options: ["paper_metadata", "paper_summaries", "paper_fulltext"]

# #     Returns:
# #         dict keyed by collection name -> list of hits
# #         each hit: {"id", "text", "meta", "distance"}
# #     """
# #     results: Dict[str, List[Dict]] = {}
# #     for name, coll in _pick_collections(scope):
# #         try:
# #             results[name] = _do_query(coll, question, k)
# #         except Exception as e:
# #             # Surface a structured error while letting callers continue with others
# #             results[name] = [{"id": None, "text": "", "meta": {"error": str(e)}, "distance": None}]
# #     return results


# # def query_chroma_merged(
# #     question: str,
# #     k_per_collection: int = 5,
# #     scope: Optional[List[CollectionName]] = None,
# #     dedupe_on_text: bool = True,
# # ) -> List[Dict]:
# #     """
# #     Convenience: query multiple collections and return a single merged list.
# #     Preserves a `source` field with the originating collection name.

# #     Dedupe (optional): removes exact duplicate `text` strings keeping the best (lowest) distance.

# #     Returns:
# #         list of {"source","id","text","meta","distance"}
# #     """
# #     scoped = scope or ["paper_metadata", "paper_summaries"]
# #     bucketed = query_chroma(question, k=k_per_collection, scope=scoped)

# #     merged: List[Dict] = []
# #     for source, hits in bucketed.items():
# #         for h in hits:
# #             merged.append({
# #                 "source": source,
# #                 **h
# #             })

# #     if dedupe_on_text and merged:
# #         best_by_text: Dict[str, Dict] = {}
# #         for h in merged:
# #             txt = h.get("text", "")
# #             dist = h.get("distance")
# #             if txt not in best_by_text:
# #                 best_by_text[txt] = h
# #             else:
# #                 prev = best_by_text[txt]
# #                 # Keep the closer (smaller distance) if available, otherwise keep first
# #                 if dist is not None and prev.get("distance") is not None:
# #                     if dist < prev["distance"]:
# #                         best_by_text[txt] = h
# #         merged = list(best_by_text.values())

# #     # Sort by distance ascending when available; otherwise preserve insertion order
# #     merged.sort(key=lambda x: (x["distance"] is None, x["distance"] if x["distance"] is not None else 0.0))
# #     return merged


# # if __name__ == "__main__":
# #     # quick sanity check
# #     print("Chroma retriever sanity check")
# #     print(f"Store: {config.CHROMA_DIR}")
# #     print(f"Model: {config.SENTENCE_TFORMER}")
# #     q = "Who studied theta oscillations and memory?"
# #     hits = query_chroma_merged(q, k_per_collection=3)
# #     for i, h in enumerate(hits, 1):
# #         meta = h.get("meta", {})
# #         title = meta.get("title") or meta.get("file_name") or ""
# #         print(f"[{i}] {h['source']} dist={h.get('distance')}: {title[:80]} | {h['text'][:120].replace('\\n',' ')}")

# # chroma_retriever.py
# import chromadb
# from chromadb.utils import embedding_functions
# import config_full as config

# # Init Chroma client
# client = chromadb.PersistentClient(path=config.CHROMA_DIR)

# # Use same embedding model as ingestion
# embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name="intfloat/e5-base-v2"
# )

# # Single unified collection
# collection = client.get_or_create_collection(
#     name="papers_all",
#     embedding_function=embedder
# )

# def query_chroma(query: str, n: int = 5):
#     """Query the unified papers_all collection."""
#     res = collection.query(
#         query_texts=[query],
#         n_results=n,
#         include=["documents", "metadatas"]
#     )
#     docs = res["documents"][0]
#     metas = res["metadatas"][0]
#     return list(zip(docs, metas))


# chroma_retriever.py
import chromadb
from chromadb.utils import embedding_functions
import config_full as config

# --- Init ---
client = chromadb.PersistentClient(path=config.CHROMA_DIR)

# ✅ Same embedder as ingestion
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/e5-base-v2"
)

collection = client.get_or_create_collection(
    name="papers_all",
    embedding_function=embedder
)

# --- Retriever ---
def query_chroma(question: str, k: int = 5, threshold: float = 0.4):
    """
    Query Chroma collection and return (doc, metadata) pairs
    filtered by cosine distance threshold.
    """
    results = collection.query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    filtered = [
        (doc, meta) for doc, meta, dist in zip(docs, metas, dists) if dist < threshold
    ]

    return filtered
