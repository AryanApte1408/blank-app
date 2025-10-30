# # # # # # # # # # # # #hybrid_langchain retriever.py
# # # # # # # # # # # # import hashlib
# # # # # # # # # # # # import chromadb
# # # # # # # # # # # # from langchain_chroma import Chroma
# # # # # # # # # # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # # # # # # # # # import torch
# # # # # # # # # # # # import config_full as config
# # # # # # # # # # # # from graph_retriever import get_papers_by_researcher, get_paper_neighbors

# # # # # # # # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # # # # # # # embed = HuggingFaceEmbeddings(
# # # # # # # # # # # #     model_name="intfloat/e5-base-v2",
# # # # # # # # # # # #     model_kwargs={"device": device}
# # # # # # # # # # # # )

# # # # # # # # # # # # client = chromadb.PersistentClient(path=config.CHROMA_DIR)
# # # # # # # # # # # # chroma = Chroma(
# # # # # # # # # # # #     client=client,
# # # # # # # # # # # #     collection_name="papers_all",
# # # # # # # # # # # #     embedding_function=embed
# # # # # # # # # # # # )

# # # # # # # # # # # # def _dedupe(lst):
# # # # # # # # # # # #     seen, out = set(), []
# # # # # # # # # # # #     for x in lst:
# # # # # # # # # # # #         h = hashlib.sha1(x.encode()).hexdigest()
# # # # # # # # # # # #         if h not in seen:
# # # # # # # # # # # #             seen.add(h)
# # # # # # # # # # # #             out.append(x)
# # # # # # # # # # # #     return out

# # # # # # # # # # # # def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8):
# # # # # # # # # # # #     # Step 1: Chroma
# # # # # # # # # # # #     chroma_docs = chroma.similarity_search(query, k=k_chroma)

# # # # # # # # # # # #     # Step 2: Extract candidates
# # # # # # # # # # # #     titles = {d.metadata.get("title") for d in chroma_docs if d.metadata.get("title")}
# # # # # # # # # # # #     researchers = {d.metadata.get("researcher") for d in chroma_docs if d.metadata.get("researcher")}

# # # # # # # # # # # #     # Step 3: Query Neo4j for enrichment
# # # # # # # # # # # #     graph_hits = []
# # # # # # # # # # # #     for r in researchers:
# # # # # # # # # # # #         graph_hits.extend(get_papers_by_researcher(r, k=k_graph))
# # # # # # # # # # # #     for t in titles:
# # # # # # # # # # # #         graph_hits.extend(get_paper_neighbors(t, k_authors=5))

# # # # # # # # # # # #     # Step 4: Fuse
# # # # # # # # # # # #     fused = []
# # # # # # # # # # # #     for d in chroma_docs:
# # # # # # # # # # # #         fused.append(f"[Chroma] {d.metadata.get('title','Untitled')} ({d.metadata.get('publication_date','?')})\n{d.page_content[:300]}")
# # # # # # # # # # # #     for g in graph_hits:
# # # # # # # # # # # #         fused.append(f"[Graph] {g.get('title','Untitled')} ({g.get('year','?')}) — {', '.join(g.get('authors', []))}")

# # # # # # # # # # # #     return {
# # # # # # # # # # # #         "graph_hits": graph_hits,
# # # # # # # # # # # #         "chroma_ctx": [d.page_content for d in chroma_docs],
# # # # # # # # # # # #         "fused_text_blocks": _dedupe(fused),
# # # # # # # # # # # #     }


# # # # # # # # # # # # hybrid_langchain_retriever.py
# # # # # # # # # # # import hashlib
# # # # # # # # # # # import torch
# # # # # # # # # # # import chromadb
# # # # # # # # # # # from langchain_chroma import Chroma
# # # # # # # # # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # # # # # # # # import config_full as config
# # # # # # # # # # # from graph_retriever import search_graph

# # # # # # # # # # # # ------------------ device + embedder ------------------
# # # # # # # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # # # # # # embed = HuggingFaceEmbeddings(
# # # # # # # # # # #     model_name="intfloat/e5-base-v2",
# # # # # # # # # # #     model_kwargs={"device": device}
# # # # # # # # # # # )

# # # # # # # # # # # # ------------------ chroma setup ------------------
# # # # # # # # # # # client = chromadb.PersistentClient(path=config.CHROMA_DIR)
# # # # # # # # # # # chroma = Chroma(
# # # # # # # # # # #     client=client,
# # # # # # # # # # #     collection_name="papers_all",
# # # # # # # # # # #     embedding_function=embed
# # # # # # # # # # # )

# # # # # # # # # # # # ------------------ helpers ------------------
# # # # # # # # # # # def _dedupe(lst):
# # # # # # # # # # #     """Deduplicate list preserving order."""
# # # # # # # # # # #     seen, out = set(), []
# # # # # # # # # # #     for x in lst:
# # # # # # # # # # #         h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
# # # # # # # # # # #         if h not in seen:
# # # # # # # # # # #             seen.add(h)
# # # # # # # # # # #             out.append(x)
# # # # # # # # # # #     return out

# # # # # # # # # # # def _safe_str(val):
# # # # # # # # # # #     if val is None:
# # # # # # # # # # #         return ""
# # # # # # # # # # #     return str(val).strip()

# # # # # # # # # # # # ------------------ hybrid search ------------------
# # # # # # # # # # # def hybrid_search(query: str, k_chroma: int = 10, k_graph: int = 8):
# # # # # # # # # # #     """
# # # # # # # # # # #     1. Retrieve top-k docs from Chroma (semantic similarity)
# # # # # # # # # # #     2. For each retrieved doc, extract researcher/title
# # # # # # # # # # #     3. Query Neo4j for related works using top keywords
# # # # # # # # # # #     4. Merge, clean, and dedupe the results
# # # # # # # # # # #     """

# # # # # # # # # # #     # ---- Step 1: Chroma retrieval ----
# # # # # # # # # # #     chroma_docs = chroma.similarity_search(query, k=k_chroma)
# # # # # # # # # # #     if not chroma_docs:
# # # # # # # # # # #         return {
# # # # # # # # # # #             "graph_hits": [],
# # # # # # # # # # #             "chroma_ctx": [],
# # # # # # # # # # #             "fused_text_blocks": ["No Chroma context found."]
# # # # # # # # # # #         }

# # # # # # # # # # #     chroma_texts = [d.page_content for d in chroma_docs if d.page_content]
# # # # # # # # # # #     chroma_metas = [d.metadata for d in chroma_docs]

# # # # # # # # # # #     # ---- Step 2: derive search hints from Chroma metadata ----
# # # # # # # # # # #     hint_titles = [_safe_str(m.get("title", "")) for m in chroma_metas if m.get("title")]
# # # # # # # # # # #     hint_researchers = [_safe_str(m.get("researcher", "")) for m in chroma_metas if m.get("researcher")]
# # # # # # # # # # #     expanded_queries = _dedupe([query] + hint_titles + hint_researchers)

# # # # # # # # # # #     # ---- Step 3: Query Neo4j with top-hints ----
# # # # # # # # # # #     graph_hits = []
# # # # # # # # # # #     for subq in expanded_queries[:3]:  # top 3 to keep speed manageable
# # # # # # # # # # #         res = search_graph(subq, k=k_graph)
# # # # # # # # # # #         graph_hits.extend(res)
# # # # # # # # # # #     # flatten + unique
# # # # # # # # # # #     unique_titles = set()
# # # # # # # # # # #     final_graph_hits = []
# # # # # # # # # # #     for g in graph_hits:
# # # # # # # # # # #         title = _safe_str(g.get("title"))
# # # # # # # # # # #         if title and title not in unique_titles:
# # # # # # # # # # #             unique_titles.add(title)
# # # # # # # # # # #             final_graph_hits.append(g)

# # # # # # # # # # #     # ---- Step 4: fuse Chroma + Neo4j ----
# # # # # # # # # # #     fused = []
# # # # # # # # # # #     for g in final_graph_hits:
# # # # # # # # # # #         fused.append(f"[Graph] {g.get('title', 'Untitled')} ({g.get('year', 'N/A')}) "
# # # # # # # # # # #                      f"— Researcher: {g.get('researcher', 'Unknown')}, "
# # # # # # # # # # #                      f"Authors: {', '.join(g.get('authors', []) or [])}")

# # # # # # # # # # #     for d in chroma_docs:
# # # # # # # # # # #         meta = d.metadata or {}
# # # # # # # # # # #         fused.append(f"[Chroma] Title: {meta.get('title', 'Untitled')} | "
# # # # # # # # # # #                      f"Researcher: {meta.get('researcher', 'Unknown')} | "
# # # # # # # # # # #                      f"Excerpt: {d.page_content[:200]}")

# # # # # # # # # # #     fused_clean = [x for x in _dedupe(fused) if "N/A" not in x and len(x.strip()) > 20]

# # # # # # # # # # #     return {
# # # # # # # # # # #         "graph_hits": final_graph_hits,
# # # # # # # # # # #         "chroma_ctx": chroma_texts,
# # # # # # # # # # #         "fused_text_blocks": fused_clean,
# # # # # # # # # # #     }

# # # # # # # # # # # # ------------------ standalone test ------------------
# # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # #     q = "papers by Gillian Youngs"
# # # # # # # # # # #     res = hybrid_search(q)
# # # # # # # # # # #     print(f"\n🔍 Query: {q}")
# # # # # # # # # # #     print(f"Chroma docs: {len(res['chroma_ctx'])}, Graph hits: {len(res['graph_hits'])}")
# # # # # # # # # # #     print("\nSample fused context:\n")
# # # # # # # # # # #     for s in res["fused_text_blocks"][:10]:
# # # # # # # # # # #         print("-", s)

# # # # # # # # # # # hybrid_langchain_retriever.py
# # # # # # # # # # import hashlib
# # # # # # # # # # import torch
# # # # # # # # # # import chromadb
# # # # # # # # # # from langchain_chroma import Chroma
# # # # # # # # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # # # # # # # import config_full as config
# # # # # # # # # # from graph_retriever import search_graph

# # # # # # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"

# # # # # # # # # # embed = HuggingFaceEmbeddings(
# # # # # # # # # #     model_name="intfloat/e5-base-v2",
# # # # # # # # # #     model_kwargs={"device": device}
# # # # # # # # # # )

# # # # # # # # # # client = chromadb.PersistentClient(path=config.CHROMA_DIR)
# # # # # # # # # # chroma = Chroma(
# # # # # # # # # #     client=client,
# # # # # # # # # #     collection_name="papers_all",
# # # # # # # # # #     embedding_function=embed
# # # # # # # # # # )

# # # # # # # # # # # ------------------ helpers ------------------
# # # # # # # # # # def _dedupe(lst):
# # # # # # # # # #     seen, out = set(), []
# # # # # # # # # #     for x in lst:
# # # # # # # # # #         h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
# # # # # # # # # #         if h not in seen:
# # # # # # # # # #             seen.add(h)
# # # # # # # # # #             out.append(x)
# # # # # # # # # #     return out

# # # # # # # # # # def _safe_str(v):
# # # # # # # # # #     return str(v).strip() if v else ""

# # # # # # # # # # # ------------------ hybrid retrieval ------------------
# # # # # # # # # # def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 5):
# # # # # # # # # #     """Chroma → Graph → merge."""
# # # # # # # # # #     chroma_docs = chroma.similarity_search(query, k=k_chroma)
# # # # # # # # # #     if not chroma_docs:
# # # # # # # # # #         return {"graph_hits": [], "chroma_ctx": [], "fused_text_blocks": ["No Chroma matches found."]}

# # # # # # # # # #     chroma_texts = [d.page_content for d in chroma_docs if d.page_content]
# # # # # # # # # #     chroma_metas = [d.metadata for d in chroma_docs]

# # # # # # # # # #     # graph enrichment based on retrieved chroma docs
# # # # # # # # # #     graph_hits = []
# # # # # # # # # #     for meta in chroma_metas:
# # # # # # # # # #         title = _safe_str(meta.get("title"))
# # # # # # # # # #         researcher = _safe_str(meta.get("researcher"))
# # # # # # # # # #         doi = _safe_str(meta.get("doi"))
# # # # # # # # # #         q = title or researcher or doi
# # # # # # # # # #         if q:
# # # # # # # # # #             graph_hits.extend(search_graph(q, k=k_graph))

# # # # # # # # # #     # ------------------ fuse ------------------
# # # # # # # # # #     fused = []
# # # # # # # # # #     for d in chroma_docs:
# # # # # # # # # #         m = d.metadata or {}
# # # # # # # # # #         fused.append(
# # # # # # # # # #             f"[Chroma] Title: {m.get('title','Untitled')} | "
# # # # # # # # # #             f"Researcher: {m.get('researcher','Unknown')} | "
# # # # # # # # # #             f"Excerpt: {d.page_content[:250]}"
# # # # # # # # # #         )

# # # # # # # # # #     for g in graph_hits:
# # # # # # # # # #         fused.append(
# # # # # # # # # #             f"[Graph] {g.get('title','Untitled')} ({g.get('year','N/A')}) — "
# # # # # # # # # #             f"Researcher: {_safe_str(g.get('researcher'))}, "
# # # # # # # # # #             f"Authors: {', '.join(g.get('authors', []) or [])}"
# # # # # # # # # #         )

# # # # # # # # # #     fused_clean = [x for x in _dedupe(fused) if len(x.strip()) > 20]
# # # # # # # # # #     return {"graph_hits": graph_hits, "chroma_ctx": chroma_texts, "fused_text_blocks": fused_clean}

# # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # #     res = hybrid_search("jeff")
# # # # # # # # # #     print(f"Chroma={len(res['chroma_ctx'])}, Graph={len(res['graph_hits'])}")
# # # # # # # # # #     for s in res["fused_text_blocks"][:8]:
# # # # # # # # # #         print("-", s)

# # # # # # # # # # hybrid_langchain_retriever.py
# # # # # # # # # import hashlib, torch, chromadb
# # # # # # # # # from langchain_chroma import Chroma
# # # # # # # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # # # # # # import config_full as config
# # # # # # # # # from graph_retriever import search_graph_from_chroma_meta

# # # # # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # # # # embed = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2", model_kwargs={"device": device})
# # # # # # # # # client = chromadb.PersistentClient(path=config.CHROMA_DIR)
# # # # # # # # # chroma = Chroma(client=client, collection_name="papers_all", embedding_function=embed)

# # # # # # # # # def _dedupe(lst):
# # # # # # # # #     seen, out = set(), []
# # # # # # # # #     for x in lst:
# # # # # # # # #         h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
# # # # # # # # #         if h not in seen:
# # # # # # # # #             seen.add(h); out.append(x)
# # # # # # # # #     return out

# # # # # # # # # def _safe_str(v): return str(v).strip() if v else ""

# # # # # # # # # def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8):
# # # # # # # # #     chroma_docs = chroma.similarity_search(query, k=k_chroma)
# # # # # # # # #     if not chroma_docs:
# # # # # # # # #         return {"graph_hits": [], "chroma_ctx": [], "fused_text_blocks": ["No Chroma matches found."]}

# # # # # # # # #     chroma_texts = [d.page_content for d in chroma_docs if d.page_content]
# # # # # # # # #     chroma_metas = [d.metadata for d in chroma_docs]

# # # # # # # # #     # Graph expansion *only for those Chroma-identified papers/researchers*
# # # # # # # # #     graph_hits = search_graph_from_chroma_meta(query, chroma_metas, k=k_graph)

# # # # # # # # #     fused = []
# # # # # # # # #     for d in chroma_docs:
# # # # # # # # #         m = d.metadata or {}
# # # # # # # # #         fused.append(
# # # # # # # # #             f"[Chroma] Title: {m.get('title','Untitled')} | Researcher: {m.get('researcher','Unknown')} | "
# # # # # # # # #             f"Excerpt: {d.page_content[:250]}"
# # # # # # # # #         )

# # # # # # # # #     for g in graph_hits:
# # # # # # # # #         fused.append(
# # # # # # # # #             f"[Graph] {g.get('title','Untitled')} ({g.get('year','N/A')}) — "
# # # # # # # # #             f"Researcher: {_safe_str(g.get('researcher'))}, "
# # # # # # # # #             f"Authors: {', '.join(g.get('authors',[]) or [])} | "
# # # # # # # # #             f"Related: {', '.join(g.get('related',[])[:3])} | "
# # # # # # # # #             f"Score: {g['score']}"
# # # # # # # # #         )

# # # # # # # # #     fused_clean = [x for x in _dedupe(fused) if len(x.strip()) > 20]
# # # # # # # # #     return {"graph_hits": graph_hits, "chroma_ctx": chroma_texts, "fused_text_blocks": fused_clean}

# # # # # # # # """
# # # # # # # # hybrid_langchain_retriever.py - Updated for enriched metadata
# # # # # # # # """
# # # # # # # # import hashlib, torch, chromadb
# # # # # # # # from langchain_chroma import Chroma
# # # # # # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # # # # # import config_full as config
# # # # # # # # from graph_retriever import search_graph_from_chroma_meta

# # # # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # # # embed = HuggingFaceEmbeddings(
# # # # # # # #     model_name="intfloat/e5-base-v2", 
# # # # # # # #     model_kwargs={"device": device}
# # # # # # # # )

# # # # # # # # # Global clients (initialized per mode)
# # # # # # # # _chroma_clients = {}


# # # # # # # # def get_chroma_client(mode="full"):
# # # # # # # #     """Get or create ChromaDB client for specific mode."""
# # # # # # # #     if mode not in _chroma_clients:
# # # # # # # #         try:
# # # # # # # #             chroma_dir = config.CHROMA_DIR_ABSTRACTS if mode == "abstracts" else config.CHROMA_DIR_FULL
# # # # # # # #             collection_name = "abstracts_all" if mode == "abstracts" else "papers_all"
            
# # # # # # # #             client = chromadb.PersistentClient(path=chroma_dir)
# # # # # # # #             _chroma_clients[mode] = Chroma(
# # # # # # # #                 client=client,
# # # # # # # #                 collection_name=collection_name,
# # # # # # # #                 embedding_function=embed
# # # # # # # #             )
# # # # # # # #             print(f"✅ ChromaDB initialized for '{mode}' mode")
# # # # # # # #         except Exception as e:
# # # # # # # #             print(f"❌ ChromaDB initialization failed for '{mode}': {e}")
# # # # # # # #             _chroma_clients[mode] = None
    
# # # # # # # #     return _chroma_clients[mode]


# # # # # # # # def _dedupe(lst):
# # # # # # # #     """Deduplicate while preserving order."""
# # # # # # # #     seen, out = set(), []
# # # # # # # #     for x in lst:
# # # # # # # #         h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
# # # # # # # #         if h not in seen:
# # # # # # # #             seen.add(h)
# # # # # # # #             out.append(x)
# # # # # # # #     return out


# # # # # # # # def _safe_str(v):
# # # # # # # #     return str(v).strip() if v else ""


# # # # # # # # def _safe_int(v):
# # # # # # # #     """Convert to int safely."""
# # # # # # # #     try:
# # # # # # # #         return int(v) if v else 0
# # # # # # # #     except (ValueError, TypeError):
# # # # # # # #         return 0


# # # # # # # # def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8, mode: str = "full"):
# # # # # # # #     """
# # # # # # # #     Hybrid retrieval with mode-aware collection switching.
# # # # # # # #     Now includes enriched metadata from abstracts_only.db
# # # # # # # #     """
    
# # # # # # # #     chroma = get_chroma_client(mode)
# # # # # # # #     chroma_docs = []
# # # # # # # #     chroma_metas = []
    
# # # # # # # #     if chroma:
# # # # # # # #         try:
# # # # # # # #             chroma_docs = chroma.similarity_search(query, k=k_chroma)
# # # # # # # #             chroma_metas = [d.metadata for d in chroma_docs if d.metadata]
            
# # # # # # # #             # Sort by year (recent first)
# # # # # # # #             chroma_docs_sorted = sorted(
# # # # # # # #                 chroma_docs,
# # # # # # # #                 key=lambda d: _safe_int(d.metadata.get("year", 0)),
# # # # # # # #                 reverse=True
# # # # # # # #             )
# # # # # # # #             chroma_docs = chroma_docs_sorted
            
# # # # # # # #         except Exception as e:
# # # # # # # #             print(f"❌ ChromaDB search failed: {e}")
    
# # # # # # # #     if not chroma_docs:
# # # # # # # #         return {
# # # # # # # #             "graph_hits": [],
# # # # # # # #             "chroma_ctx": [],
# # # # # # # #             "fused_text_blocks": [f"No results found in '{mode}' mode."]
# # # # # # # #         }
    
# # # # # # # #     # Graph expansion
# # # # # # # #     graph_hits = []
# # # # # # # #     try:
# # # # # # # #         graph_hits = search_graph_from_chroma_meta(query, chroma_metas, k=k_graph, mode=mode)
# # # # # # # #     except Exception as e:
# # # # # # # #         print(f"❌ Graph search failed: {e}")
    
# # # # # # # #     # Build fused context blocks with enriched metadata
# # # # # # # #     fused = []
    
# # # # # # # #     for d in chroma_docs:
# # # # # # # #         m = d.metadata or {}
# # # # # # # #         year = _safe_int(m.get('year', 0))
# # # # # # # #         year_str = f"({year})" if year > 0 else "(N/A)"
        
# # # # # # # #         source_label = "Abstract" if mode == "abstracts" else "Full Text"
# # # # # # # #         researcher = m.get('researcher', 'Unknown')
# # # # # # # #         source_api = m.get('source', '')  # crossref/openalex/arxiv
        
# # # # # # # #         # Include more metadata for abstracts mode
# # # # # # # #         if mode == "abstracts":
# # # # # # # #             fused.append(
# # # # # # # #                 f"[{source_label}] {year_str} {m.get('title','Untitled')}\n"
# # # # # # # #                 f"  Researcher: {researcher} | API Source: {source_api}\n"
# # # # # # # #                 f"  DOI: {m.get('doi', 'N/A')}\n"
# # # # # # # #                 f"  Excerpt: {d.page_content[:250]}"
# # # # # # # #             )
# # # # # # # #         else:
# # # # # # # #             fused.append(
# # # # # # # #                 f"[{source_label}] {year_str} Title: {m.get('title','Untitled')} | "
# # # # # # # #                 f"Researcher: {researcher} | "
# # # # # # # #                 f"Excerpt: {d.page_content[:300]}"
# # # # # # # #             )
    
# # # # # # # #     # Graph context
# # # # # # # #     for g in graph_hits:
# # # # # # # #         year = _safe_int(g.get('year', 0))
# # # # # # # #         year_str = f"({year})" if year > 0 else "(N/A)"
        
# # # # # # # #         authors = g.get('authors', []) or []
# # # # # # # #         source = g.get('source', '')
        
# # # # # # # #         if mode == "abstracts":
# # # # # # # #             fused.append(
# # # # # # # #                 f"[Graph] {year_str} {g.get('title','Untitled')}\n"
# # # # # # # #                 f"  Researcher: {_safe_str(g.get('researcher'))} | Source: {source}\n"
# # # # # # # #                 f"  Score: {g.get('score', 0)}"
# # # # # # # #             )
# # # # # # # #         else:
# # # # # # # #             fused.append(
# # # # # # # #                 f"[Graph] {year_str} {g.get('title','Untitled')} — "
# # # # # # # #                 f"Researcher: {_safe_str(g.get('researcher'))}, "
# # # # # # # #                 f"Authors: {', '.join(authors[:5])} | "
# # # # # # # #                 f"Score: {g.get('score', 0)}"
# # # # # # # #             )
    
# # # # # # # #     fused_clean = [x for x in _dedupe(fused) if len(x.strip()) > 20]
# # # # # # # #     chroma_texts = [d.page_content[:500] for d in chroma_docs[:k_chroma]]
    
# # # # # # # #     return {
# # # # # # # #         "graph_hits": graph_hits,
# # # # # # # #         "chroma_ctx": chroma_texts,
# # # # # # # #         "fused_text_blocks": fused_clean
# # # # # # # #     }
# # # # # # # """
# # # # # # # hybrid_langchain_retriever.py - Database-agnostic hybrid retrieval
# # # # # # # """
# # # # # # # import hashlib
# # # # # # # import torch
# # # # # # # import chromadb
# # # # # # # from langchain_chroma import Chroma
# # # # # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # # # # from database_manager import get_active_db_config
# # # # # # # from graph_retriever import search_graph_from_chroma_meta

# # # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # # embed = HuggingFaceEmbeddings(
# # # # # # #     model_name="intfloat/e5-base-v2", 
# # # # # # #     model_kwargs={"device": device}
# # # # # # # )

# # # # # # # # Cache of ChromaDB clients per config
# # # # # # # _chroma_clients = {}


# # # # # # # def get_chroma_client():
# # # # # # #     """Get ChromaDB client from active database config."""
# # # # # # #     config = get_active_db_config()
# # # # # # #     cache_key = f"{config.chroma_dir}_{config.chroma_collection}"
    
# # # # # # #     if cache_key not in _chroma_clients:
# # # # # # #         try:
# # # # # # #             client = chromadb.PersistentClient(path=config.chroma_dir)
# # # # # # #             _chroma_clients[cache_key] = Chroma(
# # # # # # #                 client=client,
# # # # # # #                 collection_name=config.chroma_collection,
# # # # # # #                 embedding_function=embed
# # # # # # #             )
# # # # # # #             print(f"✅ ChromaDB initialized: {config.chroma_collection}")
# # # # # # #         except Exception as e:
# # # # # # #             print(f"❌ ChromaDB initialization failed: {e}")
# # # # # # #             _chroma_clients[cache_key] = None
    
# # # # # # #     return _chroma_clients[cache_key]


# # # # # # # def clear_chroma_cache():
# # # # # # #     """Clear ChromaDB client cache (useful when switching configs)."""
# # # # # # #     global _chroma_clients
# # # # # # #     _chroma_clients = {}


# # # # # # # def _dedupe(lst):
# # # # # # #     seen, out = set(), []
# # # # # # #     for x in lst:
# # # # # # #         h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
# # # # # # #         if h not in seen:
# # # # # # #             seen.add(h)
# # # # # # #             out.append(x)
# # # # # # #     return out


# # # # # # # def _safe_str(v):
# # # # # # #     return str(v).strip() if v else ""


# # # # # # # def _safe_int(v):
# # # # # # #     try:
# # # # # # #         return int(v) if v else 0
# # # # # # #     except (ValueError, TypeError):
# # # # # # #         return 0


# # # # # # # def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8):
# # # # # # #     """
# # # # # # #     Database-agnostic hybrid retrieval.
# # # # # # #     Automatically uses active database configuration.
# # # # # # #     """
# # # # # # #     config = get_active_db_config()
# # # # # # #     chroma = get_chroma_client()
# # # # # # #     chroma_docs = []
# # # # # # #     chroma_metas = []
    
# # # # # # #     if chroma:
# # # # # # #         try:
# # # # # # #             chroma_docs = chroma.similarity_search(query, k=k_chroma)
# # # # # # #             chroma_metas = [d.metadata for d in chroma_docs if d.metadata]
            
# # # # # # #             chroma_docs_sorted = sorted(
# # # # # # #                 chroma_docs,
# # # # # # #                 key=lambda d: _safe_int(d.metadata.get("year", 0)),
# # # # # # #                 reverse=True
# # # # # # #             )
# # # # # # #             chroma_docs = chroma_docs_sorted
            
# # # # # # #         except Exception as e:
# # # # # # #             print(f"❌ ChromaDB search failed: {e}")
    
# # # # # # #     if not chroma_docs:
# # # # # # #         return {
# # # # # # #             "graph_hits": [],
# # # # # # #             "chroma_ctx": [],
# # # # # # #             "fused_text_blocks": [f"No results found in '{config.mode}' mode."]
# # # # # # #         }
    
# # # # # # #     # Graph expansion
# # # # # # #     graph_hits = []
# # # # # # #     try:
# # # # # # #         graph_hits = search_graph_from_chroma_meta(query, chroma_metas, k=k_graph)
# # # # # # #     except Exception as e:
# # # # # # #         print(f"❌ Graph search failed: {e}")
    
# # # # # # #     # Build fused context
# # # # # # #     fused = []
# # # # # # #     mode = config.mode
    
# # # # # # #     for d in chroma_docs:
# # # # # # #         m = d.metadata or {}
# # # # # # #         year = _safe_int(m.get('year', 0))
# # # # # # #         year_str = f"({year})" if year > 0 else "(N/A)"
        
# # # # # # #         source_label = "Abstract" if mode == "abstracts" else "Full Text"
# # # # # # #         researcher = m.get('researcher', 'Unknown')
# # # # # # #         source_api = m.get('source', '')
        
# # # # # # #         if mode == "abstracts":
# # # # # # #             fused.append(
# # # # # # #                 f"[{source_label}] {year_str} {m.get('title','Untitled')}\n"
# # # # # # #                 f"  Researcher: {researcher} | API Source: {source_api}\n"
# # # # # # #                 f"  DOI: {m.get('doi', 'N/A')}\n"
# # # # # # #                 f"  Excerpt: {d.page_content[:250]}"
# # # # # # #             )
# # # # # # #         else:
# # # # # # #             fused.append(
# # # # # # #                 f"[{source_label}] {year_str} Title: {m.get('title','Untitled')} | "
# # # # # # #                 f"Researcher: {researcher} | "
# # # # # # #                 f"Excerpt: {d.page_content[:300]}"
# # # # # # #             )
    
# # # # # # #     for g in graph_hits:
# # # # # # #         year = _safe_int(g.get('year', 0))
# # # # # # #         year_str = f"({year})" if year > 0 else "(N/A)"
        
# # # # # # #         authors = g.get('authors', []) or []
# # # # # # #         source = g.get('source', '')
        
# # # # # # #         if mode == "abstracts":
# # # # # # #             fused.append(
# # # # # # #                 f"[Graph] {year_str} {g.get('title','Untitled')}\n"
# # # # # # #                 f"  Researcher: {_safe_str(g.get('researcher'))} | Source: {source}\n"
# # # # # # #                 f"  Score: {g.get('score', 0)}"
# # # # # # #             )
# # # # # # #         else:
# # # # # # #             fused.append(
# # # # # # #                 f"[Graph] {year_str} {g.get('title','Untitled')} — "
# # # # # # #                 f"Researcher: {_safe_str(g.get('researcher'))}, "
# # # # # # #                 f"Authors: {', '.join(authors[:5])} | "
# # # # # # #                 f"Score: {g.get('score', 0)}"
# # # # # # #             )
    
# # # # # # #     fused_clean = [x for x in _dedupe(fused) if len(x.strip()) > 20]
# # # # # # #     chroma_texts = [d.page_content[:500] for d in chroma_docs[:k_chroma]]
    
# # # # # # #     return {
# # # # # # #         "graph_hits": graph_hits,
# # # # # # #         "chroma_ctx": chroma_texts,
# # # # # # #         "fused_text_blocks": fused_clean
# # # # # # #     }

# # # # # # """
# # # # # # hybrid_langchain_retriever.py - Works with existing ChromaDB
# # # # # # Extracts ALL information from document content
# # # # # # """
# # # # # # import hashlib
# # # # # # import torch
# # # # # # import chromadb
# # # # # # from langchain_chroma import Chroma
# # # # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # # # from database_manager import get_active_db_config
# # # # # # from graph_retriever import search_graph_from_chroma_meta_enhanced

# # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # embed = HuggingFaceEmbeddings(
# # # # # #     model_name="intfloat/e5-base-v2", 
# # # # # #     model_kwargs={"device": device}
# # # # # # )

# # # # # # _chroma_clients = {}


# # # # # # def get_chroma_client():
# # # # # #     """Get ChromaDB client from active database config."""
# # # # # #     config = get_active_db_config()
# # # # # #     cache_key = f"{config.chroma_dir}_{config.chroma_collection}"
    
# # # # # #     if cache_key not in _chroma_clients:
# # # # # #         try:
# # # # # #             client = chromadb.PersistentClient(path=config.chroma_dir)
# # # # # #             _chroma_clients[cache_key] = Chroma(
# # # # # #                 client=client,
# # # # # #                 collection_name=config.chroma_collection,
# # # # # #                 embedding_function=embed
# # # # # #             )
# # # # # #             print(f"✅ ChromaDB initialized: {config.chroma_collection}")
# # # # # #         except Exception as e:
# # # # # #             print(f"❌ ChromaDB initialization failed: {e}")
# # # # # #             _chroma_clients[cache_key] = None
    
# # # # # #     return _chroma_clients[cache_key]


# # # # # # def clear_chroma_cache():
# # # # # #     global _chroma_clients
# # # # # #     _chroma_clients = {}


# # # # # # def _dedupe(lst):
# # # # # #     seen, out = set(), []
# # # # # #     for x in lst:
# # # # # #         h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
# # # # # #         if h not in seen:
# # # # # #             seen.add(h)
# # # # # #             out.append(x)
# # # # # #     return out


# # # # # # def _safe_str(v):
# # # # # #     return str(v).strip() if v else ""


# # # # # # def _safe_int(v):
# # # # # #     try:
# # # # # #         return int(v) if v else 0
# # # # # #     except (ValueError, TypeError):
# # # # # #         return 0


# # # # # # def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8):
# # # # # #     """
# # # # # #     Hybrid search that extracts names from document content.
# # # # # #     Works with existing database!
# # # # # #     """
# # # # # #     config = get_active_db_config()
# # # # # #     chroma = get_chroma_client()
# # # # # #     chroma_docs = []
# # # # # #     chroma_metas = []
    
# # # # # #     if chroma:
# # # # # #         try:
# # # # # #             chroma_docs = chroma.similarity_search(query, k=k_chroma)
# # # # # #             chroma_metas = [d.metadata for d in chroma_docs if d.metadata]
            
# # # # # #             # Sort by year
# # # # # #             chroma_docs_sorted = sorted(
# # # # # #                 chroma_docs,
# # # # # #                 key=lambda d: _safe_int(d.metadata.get("year", 0)),
# # # # # #                 reverse=True
# # # # # #             )
# # # # # #             chroma_docs = chroma_docs_sorted
            
# # # # # #         except Exception as e:
# # # # # #             print(f"❌ ChromaDB search failed: {e}")
    
# # # # # #     if not chroma_docs:
# # # # # #         return {
# # # # # #             "graph_hits": [],
# # # # # #             "chroma_ctx": [],
# # # # # #             "fused_text_blocks": [f"No results found in '{config.mode}' mode."]
# # # # # #         }
    
# # # # # #     # Graph expansion using BOTH doc content and metadata
# # # # # #     graph_hits = []
# # # # # #     try:
# # # # # #         # Pass documents WITH metadata for enhanced extraction
# # # # # #         docs_with_meta = [(doc, doc.metadata) for doc in chroma_docs]
# # # # # #         graph_hits = search_graph_from_chroma_meta_enhanced(query, docs_with_meta, k=k_graph)
# # # # # #     except Exception as e:
# # # # # #         print(f"❌ Graph search failed: {e}")
    
# # # # # #     # Build fused context
# # # # # #     fused = []
    
# # # # # #     for d in chroma_docs:
# # # # # #         m = d.metadata or {}
# # # # # #         year = _safe_int(m.get('year', 0))
# # # # # #         year_str = f"({year})" if year > 0 else "(N/A)"
        
# # # # # #         # Extract all name-related fields dynamically
# # # # # #         names = []
# # # # # #         for key, value in m.items():
# # # # # #             if any(term in key.lower() for term in ['author', 'researcher', 'name']):
# # # # # #                 value_str = _safe_str(value)
# # # # # #                 if value_str and value_str != 'Unknown':
# # # # # #                     names.append(value_str)
        
# # # # # #         name_display = "; ".join(set(names)) if names else "Unknown"
# # # # # #         title = m.get('title', 'Untitled')
# # # # # #         source = m.get('source', '')
        
# # # # # #         fused.append(
# # # # # #             f"[ChromaDB] {year_str} {title}\n"
# # # # # #             f"  Contributors: {name_display}\n"
# # # # # #             f"  Source: {source}"
# # # # # #         )
    
# # # # # #     # Graph context
# # # # # #     for g in graph_hits:
# # # # # #         year = _safe_int(g.get('year', 0))
# # # # # #         year_str = f"({year})" if year > 0 else "(N/A)"
        
# # # # # #         authors = g.get('authors', []) or []
# # # # # #         researcher = g.get('researcher', 'Unknown')
        
# # # # # #         fused.append(
# # # # # #             f"[Graph] {year_str} {g.get('title','Untitled')}\n"
# # # # # #             f"  Researcher: {researcher}\n"
# # # # # #             f"  Co-authors: {', '.join(authors[:5])}\n"
# # # # # #             f"  Relevance: {g.get('score', 0)}"
# # # # # #         )
    
# # # # # #     fused_clean = [x for x in _dedupe(fused) if len(x.strip()) > 20]
# # # # # #     chroma_texts = [d.page_content[:500] for d in chroma_docs[:k_chroma]]
    
# # # # # #     return {
# # # # # #         "graph_hits": graph_hits,
# # # # # #         "chroma_ctx": chroma_texts,
# # # # # #         "fused_text_blocks": fused_clean
# # # # # #     }

# # # # # """
# # # # # hybrid_langchain_retriever.py - Enhanced with better relevance filtering
# # # # # """
# # # # # import hashlib
# # # # # import torch
# # # # # import re
# # # # # from typing import List, Dict, Optional
# # # # # from collections import Counter
# # # # # import chromadb
# # # # # from langchain_chroma import Chroma
# # # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # # from database_manager import get_active_db_config
# # # # # from graph_retriever import search_graph_from_chroma_meta_enhanced

# # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # embed = HuggingFaceEmbeddings(
# # # # #     model_name="intfloat/e5-base-v2", 
# # # # #     model_kwargs={"device": device}
# # # # # )

# # # # # _chroma_clients = {}


# # # # # def get_chroma_client():
# # # # #     """Get ChromaDB client from active database config."""
# # # # #     config = get_active_db_config()
# # # # #     cache_key = f"{config.chroma_dir}_{config.chroma_collection}"
    
# # # # #     if cache_key not in _chroma_clients:
# # # # #         try:
# # # # #             client = chromadb.PersistentClient(path=config.chroma_dir)
# # # # #             _chroma_clients[cache_key] = Chroma(
# # # # #                 client=client,
# # # # #                 collection_name=config.chroma_collection,
# # # # #                 embedding_function=embed
# # # # #             )
# # # # #             print(f"✅ ChromaDB initialized: {config.chroma_collection}")
# # # # #         except Exception as e:
# # # # #             print(f"❌ ChromaDB initialization failed: {e}")
# # # # #             _chroma_clients[cache_key] = None
    
# # # # #     return _chroma_clients[cache_key]


# # # # # def clear_chroma_cache():
# # # # #     global _chroma_clients
# # # # #     _chroma_clients = {}


# # # # # def _dedupe(lst):
# # # # #     seen, out = set(), []
# # # # #     for x in lst:
# # # # #         h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
# # # # #         if h not in seen:
# # # # #             seen.add(h)
# # # # #             out.append(x)
# # # # #     return out


# # # # # def _safe_str(v):
# # # # #     return str(v).strip() if v else ""


# # # # # def _safe_int(v):
# # # # #     try:
# # # # #         return int(v) if v else 0
# # # # #     except (ValueError, TypeError):
# # # # #         return 0


# # # # # def extract_query_keywords(query: str) -> List[str]:
# # # # #     """Extract important keywords from user query."""
# # # # #     # Remove common question words
# # # # #     stop_words = {
# # # # #         'does', 'anyone', 'study', 'about', 'tell', 'me', 'what', 'who', 
# # # # #         'when', 'where', 'which', 'how', 'can', 'you', 'the', 'syracuse',
# # # # #         'university', 'research', 'researchers', 'at', 'on', 'in'
# # # # #     }
    
# # # # #     # Extract words
# # # # #     words = re.findall(r'\b[A-Za-z]+\b', query.lower())
# # # # #     keywords = [w for w in words if w not in stop_words and len(w) > 3]
    
# # # # #     return keywords


# # # # # def calculate_relevance_score(doc_content: str, metadata: Dict, query_keywords: List[str]) -> float:
# # # # #     """
# # # # #     Calculate how relevant a document is to the query keywords.
# # # # #     """
# # # # #     if not query_keywords:
# # # # #         return 0.5  # Neutral score
    
# # # # #     doc_lower = doc_content.lower()
# # # # #     meta_text = " ".join(str(v) for v in metadata.values()).lower()
# # # # #     combined = doc_lower + " " + meta_text
    
# # # # #     # Count keyword occurrences
# # # # #     matches = sum(1 for kw in query_keywords if kw in combined)
    
# # # # #     # Score = percentage of keywords matched
# # # # #     score = matches / len(query_keywords)
    
# # # # #     return score


# # # # # def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8):
# # # # #     """
# # # # #     Enhanced hybrid search with better relevance filtering.
# # # # #     """
# # # # #     config = get_active_db_config()
# # # # #     chroma = get_chroma_client()
# # # # #     chroma_docs = []
    
# # # # #     # Extract keywords from query for filtering
# # # # #     query_keywords = extract_query_keywords(query)
# # # # #     print(f"🔑 Query keywords: {query_keywords}")
    
# # # # #     if chroma:
# # # # #         try:
# # # # #             # Get more results initially for filtering
# # # # #             initial_k = k_chroma * 3
# # # # #             chroma_docs_raw = chroma.similarity_search(query, k=initial_k)
            
# # # # #             # Calculate relevance scores
# # # # #             scored_docs = []
# # # # #             for doc in chroma_docs_raw:
# # # # #                 relevance = calculate_relevance_score(doc.page_content, doc.metadata, query_keywords)
# # # # #                 scored_docs.append((doc, relevance))
            
# # # # #             # Filter out low-relevance results (threshold: 0.2)
# # # # #             filtered_docs = [(doc, score) for doc, score in scored_docs if score > 0.2]
            
# # # # #             print(f"📊 ChromaDB: {len(chroma_docs_raw)} raw -> {len(filtered_docs)} after filtering")
            
# # # # #             # Sort by relevance first, then year
# # # # #             filtered_docs.sort(key=lambda x: (-x[1], -_safe_int(x[0].metadata.get("year", 0))))
            
# # # # #             # Take top k
# # # # #             chroma_docs = [doc for doc, score in filtered_docs[:k_chroma]]
            
# # # # #             if not chroma_docs:
# # # # #                 # Fallback: use top results even if low relevance
# # # # #                 chroma_docs = [doc for doc, score in scored_docs[:k_chroma]]
            
# # # # #         except Exception as e:
# # # # #             print(f"❌ ChromaDB search failed: {e}")
    
# # # # #     if not chroma_docs:
# # # # #         return {
# # # # #             "graph_hits": [],
# # # # #             "chroma_ctx": [],
# # # # #             "fused_text_blocks": [f"No results found in '{config.mode}' mode."]
# # # # #         }
    
# # # # #     # Graph expansion
# # # # #     graph_hits = []
# # # # #     try:
# # # # #         docs_with_meta = [(doc, doc.metadata) for doc in chroma_docs]
# # # # #         graph_hits = search_graph_from_chroma_meta_enhanced(query, docs_with_meta, k=k_graph)
# # # # #     except Exception as e:
# # # # #         print(f"❌ Graph search failed: {e}")
    
# # # # #     # Build fused context
# # # # #     fused = []
    
# # # # #     for d in chroma_docs:
# # # # #         m = d.metadata or {}
# # # # #         year = _safe_int(m.get('year', 0))
# # # # #         year_str = f"({year})" if year > 0 else "(N/A)"
        
# # # # #         # Extract names dynamically
# # # # #         names = []
# # # # #         for key, value in m.items():
# # # # #             if any(term in key.lower() for term in ['author', 'researcher', 'name']):
# # # # #                 value_str = _safe_str(value)
# # # # #                 if value_str and value_str not in ['Unknown', 'N/A', '']:
# # # # #                     split_names = re.split(r'[;,\|]', value_str)
# # # # #                     names.extend([n.strip() for n in split_names if n.strip()])
        
# # # # #         names = list(set(names))
# # # # #         name_display = "; ".join(names[:5]) if names else "Unknown"
        
# # # # #         title = m.get('title', 'Untitled')
# # # # #         source = m.get('source', '')
        
# # # # #         fused.append(
# # # # #             f"[ChromaDB] {year_str} {title}\n"
# # # # #             f"  Authors: {name_display}\n"
# # # # #             f"  Source: {source}"
# # # # #         )
    
# # # # #     # Graph context
# # # # #     for g in graph_hits:
# # # # #         year = _safe_int(g.get('year', 0))
# # # # #         year_str = f"({year})" if year > 0 else "(N/A)"
        
# # # # #         authors = g.get('authors', []) or []
# # # # #         researcher = g.get('researcher', 'Unknown')
        
# # # # #         fused.append(
# # # # #             f"[Graph] {year_str} {g.get('title','Untitled')}\n"
# # # # #             f"  Researcher: {researcher}\n"
# # # # #             f"  Co-authors: {', '.join(authors[:5])}\n"
# # # # #             f"  Relevance: {g.get('score', 0)}"
# # # # #         )
    
# # # # #     fused_clean = [x for x in _dedupe(fused) if len(x.strip()) > 20]
# # # # #     chroma_texts = [d.page_content[:500] for d in chroma_docs[:k_chroma]]
    
# # # # #     return {
# # # # #         "graph_hits": graph_hits,
# # # # #         "chroma_ctx": chroma_texts,
# # # # #         "fused_text_blocks": fused_clean
# # # # #     }

# # # # """
# # # # hybrid_langchain_retriever.py - ChromaDB semantic search, Neo4j for graph only
# # # # """
# # # # import hashlib
# # # # import torch
# # # # import re
# # # # from typing import List
# # # # import chromadb
# # # # from langchain_chroma import Chroma
# # # # from langchain_huggingface import HuggingFaceEmbeddings
# # # # from database_manager import get_active_db_config

# # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # embed = HuggingFaceEmbeddings(
# # # #     model_name="intfloat/e5-base-v2", 
# # # #     model_kwargs={"device": device}
# # # # )

# # # # _chroma_clients = {}


# # # # def get_chroma_client():
# # # #     config = get_active_db_config()
# # # #     cache_key = f"{config.chroma_dir}_{config.chroma_collection}"
    
# # # #     if cache_key not in _chroma_clients:
# # # #         try:
# # # #             client = chromadb.PersistentClient(path=config.chroma_dir)
# # # #             _chroma_clients[cache_key] = Chroma(
# # # #                 client=client,
# # # #                 collection_name=config.chroma_collection,
# # # #                 embedding_function=embed
# # # #             )
# # # #             print(f"✅ ChromaDB initialized: {config.chroma_collection}")
# # # #         except Exception as e:
# # # #             print(f"❌ ChromaDB initialization failed: {e}")
# # # #             _chroma_clients[cache_key] = None
    
# # # #     return _chroma_clients[cache_key]


# # # # def clear_chroma_cache():
# # # #     global _chroma_clients
# # # #     _chroma_clients = {}


# # # # def _dedupe(lst):
# # # #     seen, out = set(), []
# # # #     for x in lst:
# # # #         h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
# # # #         if h not in seen:
# # # #             seen.add(h)
# # # #             out.append(x)
# # # #     return out


# # # # def _safe_str(v):
# # # #     return str(v).strip() if v else ""


# # # # def _safe_int(v):
# # # #     try:
# # # #         return int(v) if v else 0
# # # #     except (ValueError, TypeError):
# # # #         return 0


# # # # def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8):
# # # #     """
# # # #     SIMPLIFIED FLOW:
# # # #     1. ChromaDB does semantic similarity search
# # # #     2. Take those results and just pass them to Neo4j for graph structure
# # # #     3. No searching in Neo4j - just fetch the graph for those papers
# # # #     """
# # # #     config = get_active_db_config()
# # # #     chroma = get_chroma_client()
    
# # # #     if not chroma:
# # # #         return {
# # # #             "graph_hits": [],
# # # #             "chroma_ctx": [],
# # # #             "fused_text_blocks": ["ChromaDB not available."]
# # # #         }
    
# # # #     # STEP 1: Pure semantic similarity search in ChromaDB
# # # #     print(f"🔍 ChromaDB semantic search: '{query}'")
    
# # # #     try:
# # # #         chroma_docs = chroma.similarity_search(query, k=k_chroma)
# # # #         print(f"📚 ChromaDB found {len(chroma_docs)} semantically similar papers")
        
# # # #         # Sort by year (recent first)
# # # #         chroma_docs = sorted(
# # # #             chroma_docs,
# # # #             key=lambda d: _safe_int(d.metadata.get("year", 0)),
# # # #             reverse=True
# # # #         )
        
# # # #     except Exception as e:
# # # #         print(f"❌ ChromaDB search failed: {e}")
# # # #         return {
# # # #             "graph_hits": [],
# # # #             "chroma_ctx": [],
# # # #             "fused_text_blocks": ["ChromaDB search failed."]
# # # #         }
    
# # # #     if not chroma_docs:
# # # #         return {
# # # #             "graph_hits": [],
# # # #             "chroma_ctx": [],
# # # #             "fused_text_blocks": ["No semantically similar papers found."]
# # # #         }
    
# # # #     # STEP 2: Convert ChromaDB results to graph_hits format (for Neo4j visualization)
# # # #     # Just restructure the data - NO searching in Neo4j yet
# # # #     graph_hits = []
    
# # # #     for doc in chroma_docs:
# # # #         m = doc.metadata or {}
        
# # # #         # Create graph_hit entry from ChromaDB metadata
# # # #         graph_hit = {
# # # #             'paper_id': m.get('doi', '') or m.get('paper_id', ''),
# # # #             'title': m.get('title', 'Untitled'),
# # # #             'year': _safe_int(m.get('year', 0)),
# # # #             'doi': m.get('doi', ''),
# # # #             'source': m.get('source', ''),
# # # #             'researcher': m.get('researcher', 'Unknown'),
# # # #             'authors': [],  # Will be populated from Neo4j if available
# # # #             'score': 0.0
# # # #         }
        
# # # #         # Extract authors from metadata
# # # #         for key, value in m.items():
# # # #             if 'author' in key.lower():
# # # #                 value_str = _safe_str(value)
# # # #                 if value_str and value_str not in ['Unknown', 'N/A']:
# # # #                     authors = re.split(r'[;,\|]', value_str)
# # # #                     graph_hit['authors'] = [a.strip() for a in authors if a.strip()]
# # # #                     break
        
# # # #         graph_hits.append(graph_hit)
    
# # # #     print(f"✅ Prepared {len(graph_hits)} papers for graph visualization")
    
# # # #     # STEP 3: Build context display
# # # #     fused = []
    
# # # #     for idx, d in enumerate(chroma_docs, 1):
# # # #         m = d.metadata or {}
# # # #         year = _safe_int(m.get('year', 0))
# # # #         year_str = f"({year})" if year > 0 else "(N/A)"
        
# # # #         # Extract names
# # # #         names = []
# # # #         for key, value in m.items():
# # # #             if any(term in key.lower() for term in ['author', 'researcher', 'name']):
# # # #                 value_str = _safe_str(value)
# # # #                 if value_str and value_str not in ['Unknown', 'N/A', '']:
# # # #                     split_names = re.split(r'[;,\|]', value_str)
# # # #                     names.extend([n.strip() for n in split_names if n.strip()])
        
# # # #         names = list(set(names))
# # # #         name_display = "; ".join(names[:5]) if names else "Unknown"
        
# # # #         title = m.get('title', 'Untitled')
# # # #         source = m.get('source', '')
# # # #         doi = m.get('doi', '')
        
# # # #         fused.append(
# # # #             f"{idx}. {year_str} {title}\n"
# # # #             f"   Authors: {name_display}\n"
# # # #             f"   Source: {source} | DOI: {doi}"
# # # #         )
    
# # # #     fused_clean = _dedupe(fused)
# # # #     chroma_texts = [d.page_content for d in chroma_docs]
    
# # # #     return {
# # # #         "graph_hits": graph_hits,  # Pass to graph visualizer
# # # #         "chroma_ctx": chroma_texts,  # For LLM
# # # #         "fused_text_blocks": fused_clean  # For display
# # # #     }

# # # """
# # # hybrid_langchain_retriever.py - Pure semantic ChromaDB, Neo4j for graph only
# # # """
# # # import hashlib
# # # import torch
# # # import re
# # # from typing import List
# # # import chromadb
# # # from langchain_chroma import Chroma
# # # from langchain_huggingface import HuggingFaceEmbeddings
# # # from database_manager import get_active_db_config

# # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # embed = HuggingFaceEmbeddings(
# # #     model_name="intfloat/e5-base-v2", 
# # #     model_kwargs={"device": device}
# # # )

# # # _chroma_clients = {}


# # # def get_chroma_client():
# # #     config = get_active_db_config()
# # #     cache_key = f"{config.chroma_dir}_{config.chroma_collection}"
    
# # #     if cache_key not in _chroma_clients:
# # #         try:
# # #             client = chromadb.PersistentClient(path=config.chroma_dir)
# # #             _chroma_clients[cache_key] = Chroma(
# # #                 client=client,
# # #                 collection_name=config.chroma_collection,
# # #                 embedding_function=embed
# # #             )
# # #             print(f"✅ ChromaDB initialized: {config.chroma_collection}")
# # #         except Exception as e:
# # #             print(f"❌ ChromaDB initialization failed: {e}")
# # #             _chroma_clients[cache_key] = None
    
# # #     return _chroma_clients[cache_key]


# # # def clear_chroma_cache():
# # #     global _chroma_clients
# # #     _chroma_clients = {}


# # # def _dedupe(lst):
# # #     seen, out = set(), []
# # #     for x in lst:
# # #         h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
# # #         if h not in seen:
# # #             seen.add(h)
# # #             out.append(x)
# # #     return out


# # # def _safe_str(v):
# # #     return str(v).strip() if v else ""


# # # def _safe_int(v):
# # #     try:
# # #         return int(v) if v else 0
# # #     except (ValueError, TypeError):
# # #         return 0


# # # def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8):
# # #     """
# # #     Pure semantic search:
# # #     1. ChromaDB semantic similarity (embeddings)
# # #     2. Pass results to Neo4j for graph structure
# # #     """
# # #     config = get_active_db_config()
# # #     chroma = get_chroma_client()
    
# # #     if not chroma:
# # #         return {
# # #             "graph_hits": [],
# # #             "chroma_ctx": [],
# # #             "fused_text_blocks": ["ChromaDB not available."]
# # #         }
    
# # #     # SEMANTIC SEARCH in ChromaDB
# # #     print(f"🔍 Semantic search in ChromaDB: '{query}'")
    
# # #     try:
# # #         chroma_docs = chroma.similarity_search(query, k=k_chroma)
# # #         print(f"📚 ChromaDB found {len(chroma_docs)} semantically similar papers")
        
# # #         # Sort by year
# # #         chroma_docs = sorted(
# # #             chroma_docs,
# # #             key=lambda d: _safe_int(d.metadata.get("year", 0)),
# # #             reverse=True
# # #         )
        
# # #     except Exception as e:
# # #         print(f"❌ ChromaDB search failed: {e}")
# # #         return {
# # #             "graph_hits": [],
# # #             "chroma_ctx": [],
# # #             "fused_text_blocks": ["ChromaDB search failed."]
# # #         }
    
# # #     if not chroma_docs:
# # #         return {
# # #             "graph_hits": [],
# # #             "chroma_ctx": [],
# # #             "fused_text_blocks": ["No semantically similar papers found."]
# # #         }
    
# # #     # Convert ChromaDB results to graph_hits format
# # #     graph_hits = []
    
# # #     for doc in chroma_docs:
# # #         m = doc.metadata or {}
        
# # #         # Build graph_hit from ALL available metadata
# # #         graph_hit = {}
        
# # #         # Copy ALL metadata
# # #         for key, value in m.items():
# # #             graph_hit[key] = value
        
# # #         # Ensure standard fields exist
# # #         graph_hit['paper_id'] = m.get('doi', '') or m.get('paper_id', '') or str(hash(doc.page_content))
# # #         graph_hit['title'] = m.get('title', 'Untitled')
# # #         graph_hit['year'] = _safe_int(m.get('year', 0))
# # #         graph_hit['doi'] = m.get('doi', '')
# # #         graph_hit['source'] = m.get('source', '')
# # #         graph_hit['researcher'] = m.get('researcher', 'Unknown')
        
# # #         # Extract authors from any author-related field
# # #         authors = []
# # #         for key, value in m.items():
# # #             if 'author' in key.lower():
# # #                 value_str = _safe_str(value)
# # #                 if value_str and value_str not in ['Unknown', 'N/A']:
# # #                     authors = re.split(r'[;,\|]', value_str)
# # #                     break
        
# # #         graph_hit['authors'] = [a.strip() for a in authors if a.strip()]
# # #         graph_hit['score'] = 0.0
        
# # #         graph_hits.append(graph_hit)
    
# # #     print(f"✅ Prepared {len(graph_hits)} papers for Neo4j graph visualization")
    
# # #     # Build display context
# # #     fused = []
    
# # #     for idx, d in enumerate(chroma_docs, 1):
# # #         m = d.metadata or {}
# # #         year = _safe_int(m.get('year', 0))
# # #         year_str = f"({year})" if year > 0 else "(N/A)"
        
# # #         # Extract all name fields
# # #         names = []
# # #         for key, value in m.items():
# # #             if any(term in key.lower() for term in ['author', 'researcher', 'name']):
# # #                 value_str = _safe_str(value)
# # #                 if value_str and value_str not in ['Unknown', 'N/A', '']:
# # #                     split_names = re.split(r'[;,\|]', value_str)
# # #                     names.extend([n.strip() for n in split_names if n.strip()])
        
# # #         names = list(set(names))
# # #         name_display = "; ".join(names[:5]) if names else "Unknown"
        
# # #         title = m.get('title', 'Untitled')
# # #         source = m.get('source', '')
# # #         doi = m.get('doi', '')
        
# # #         fused.append(
# # #             f"{idx}. {year_str} {title}\n"
# # #             f"   Authors: {name_display}\n"
# # #             f"   Source: {source} | DOI: {doi}"
# # #         )
    
# # #     fused_clean = _dedupe(fused)
# # #     chroma_texts = [d.page_content for d in chroma_docs]
    
# # #     return {
# # #         "graph_hits": graph_hits,
# # #         "chroma_ctx": chroma_texts,
# # #         "fused_text_blocks": fused_clean
# # #     }

# # """
# # hybrid_langchain_retriever.py - Pure semantic ChromaDB, Neo4j for graph only
# # """
# # import hashlib
# # import torch
# # import re
# # from typing import List
# # import chromadb
# # from langchain_chroma import Chroma
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from database_manager import get_active_db_config

# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # embed = HuggingFaceEmbeddings(
# #     model_name="intfloat/e5-base-v2", 
# #     model_kwargs={"device": device}
# # )

# # _chroma_clients = {}


# # def get_chroma_client():
# #     config = get_active_db_config()
# #     cache_key = f"{config.chroma_dir}_{config.chroma_collection}"
    
# #     if cache_key not in _chroma_clients:
# #         try:
# #             client = chromadb.PersistentClient(path=config.chroma_dir)
# #             _chroma_clients[cache_key] = Chroma(
# #                 client=client,
# #                 collection_name=config.chroma_collection,
# #                 embedding_function=embed
# #             )
# #             print(f"✅ ChromaDB initialized: {config.chroma_collection}")
# #         except Exception as e:
# #             print(f"❌ ChromaDB initialization failed: {e}")
# #             _chroma_clients[cache_key] = None
    
# #     return _chroma_clients[cache_key]


# # def clear_chroma_cache():
# #     global _chroma_clients
# #     _chroma_clients = {}


# # def _dedupe(lst):
# #     seen, out = set(), []
# #     for x in lst:
# #         h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
# #         if h not in seen:
# #             seen.add(h)
# #             out.append(x)
# #     return out


# # def _safe_str(v):
# #     return str(v).strip() if v else ""


# # def _safe_int(v):
# #     try:
# #         return int(v) if v else 0
# #     except (ValueError, TypeError):
# #         return 0


# # def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8):
# #     """
# #     Pure semantic search:
# #     1. ChromaDB semantic similarity (embeddings)
# #     2. Pass results to Neo4j for graph structure
# #     """
# #     config = get_active_db_config()
# #     chroma = get_chroma_client()
    
# #     if not chroma:
# #         return {
# #             "graph_hits": [],
# #             "chroma_ctx": [],
# #             "fused_text_blocks": ["ChromaDB not available."]
# #         }
    
# #     # SEMANTIC SEARCH in ChromaDB
# #     print(f"🔍 Semantic search in ChromaDB: '{query}'")
    
# #     try:
# #         chroma_docs = chroma.similarity_search(query, k=k_chroma)
# #         print(f"📚 ChromaDB found {len(chroma_docs)} semantically similar papers")
        
# #         # Sort by year
# #         chroma_docs = sorted(
# #             chroma_docs,
# #             key=lambda d: _safe_int(d.metadata.get("year", 0)),
# #             reverse=True
# #         )
        
# #     except Exception as e:
# #         print(f"❌ ChromaDB search failed: {e}")
# #         return {
# #             "graph_hits": [],
# #             "chroma_ctx": [],
# #             "fused_text_blocks": ["ChromaDB search failed."]
# #         }
    
# #     if not chroma_docs:
# #         return {
# #             "graph_hits": [],
# #             "chroma_ctx": [],
# #             "fused_text_blocks": ["No semantically similar papers found."]
# #         }
    
# #     # Convert ChromaDB results to graph_hits format
# #     graph_hits = []
    
# #     for doc in chroma_docs:
# #         m = doc.metadata or {}
        
# #         # Build graph_hit from ALL available metadata
# #         graph_hit = {}
        
# #         # Copy ALL metadata
# #         for key, value in m.items():
# #             graph_hit[key] = value
        
# #         # Ensure standard fields exist
# #         graph_hit['paper_id'] = m.get('doi', '') or m.get('paper_id', '') or str(hash(doc.page_content))
# #         graph_hit['title'] = m.get('title', 'Untitled')
# #         graph_hit['year'] = _safe_int(m.get('year', 0))
# #         graph_hit['doi'] = m.get('doi', '')
# #         graph_hit['source'] = m.get('source', '')
# #         graph_hit['researcher'] = m.get('researcher', 'Unknown')
        
# #         # Extract authors from any author-related field
# #         authors = []
# #         for key, value in m.items():
# #             if 'author' in key.lower():
# #                 value_str = _safe_str(value)
# #                 if value_str and value_str not in ['Unknown', 'N/A']:
# #                     authors = re.split(r'[;,\|]', value_str)
# #                     break
        
# #         graph_hit['authors'] = [a.strip() for a in authors if a.strip()]
# #         graph_hit['score'] = 0.0
        
# #         graph_hits.append(graph_hit)
    
# #     print(f"✅ Prepared {len(graph_hits)} papers for Neo4j graph visualization")
    
# #     # Build display context
# #     fused = []
    
# #     for idx, d in enumerate(chroma_docs, 1):
# #         m = d.metadata or {}
# #         year = _safe_int(m.get('year', 0))
# #         year_str = f"({year})" if year > 0 else "(N/A)"
        
# #         # Extract all name fields
# #         names = []
# #         for key, value in m.items():
# #             if any(term in key.lower() for term in ['author', 'researcher', 'name']):
# #                 value_str = _safe_str(value)
# #                 if value_str and value_str not in ['Unknown', 'N/A', '']:
# #                     split_names = re.split(r'[;,\|]', value_str)
# #                     names.extend([n.strip() for n in split_names if n.strip()])
        
# #         names = list(set(names))
# #         name_display = "; ".join(names[:5]) if names else "Unknown"
        
# #         title = m.get('title', 'Untitled')
# #         source = m.get('source', '')
# #         doi = m.get('doi', '')
        
# #         fused.append(
# #             f"{idx}. {year_str} {title}\n"
# #             f"   Authors: {name_display}\n"
# #             f"   Source: {source} | DOI: {doi}"
# #         )
    
# #     fused_clean = _dedupe(fused)
# #     chroma_texts = [d.page_content for d in chroma_docs]
    
# #     return {
# #         "graph_hits": graph_hits,
# #         "chroma_ctx": chroma_texts,
# #         "fused_text_blocks": fused_clean
# #     }

# """
# hybrid_langchain_retriever.py - Pure semantic ChromaDB, Neo4j for graph only
# (Updated Nov 2025)
# - DB-level sort: apply newest → oldest using database_manager.sort_fused_result_newest_first
# - Keeps year-desc sort of Chroma results as a fast-path; DB layer enforces final order
# """
# import hashlib
# import torch
# import re
# from typing import List
# import chromadb
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from database_manager import get_active_db_config
# # NEW: import the sorter from DB layer
# from database_manager import sort_fused_result_newest_first

# device = "cuda" if torch.cuda.is_available() else "cpu"
# embed = HuggingFaceEmbeddings(
#     model_name="intfloat/e5-base-v2", 
#     model_kwargs={"device": device}
# )

# _chroma_clients = {}


# def get_chroma_client():
#     config = get_active_db_config()
#     cache_key = f"{config.chroma_dir}_{config.chroma_collection}"
    
#     if cache_key not in _chroma_clients:
#         try:
#             client = chromadb.PersistentClient(path=config.chroma_dir)
#             _chroma_clients[cache_key] = Chroma(
#                 client=client,
#                 collection_name=config.chroma_collection,
#                 embedding_function=embed
#             )
#             print(f"✅ ChromaDB initialized: {config.chroma_collection}")
#         except Exception as e:
#             print(f"❌ ChromaDB initialization failed: {e}")
#             _chroma_clients[cache_key] = None
    
#     return _chroma_clients[cache_key]


# def clear_chroma_cache():
#     global _chroma_clients
#     _chroma_clients = {}


# def _dedupe(lst):
#     seen, out = set(), []
#     for x in lst:
#         h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
#         if h not in seen:
#             seen.add(h)
#             out.append(x)
#     return out


# def _safe_str(v):
#     return str(v).strip() if v else ""


# def _safe_int(v):
#     try:
#         return int(v) if v else 0
#     except (ValueError, TypeError):
#         return 0


# def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8):
#     """
#     Pure semantic search:
#     1. ChromaDB semantic similarity (embeddings)
#     2. Prepare Neo4j graph hits for visualization (no search logic here)
#     3. DB-layer sort (newest → oldest) before returning fused blocks
#     """
#     config = get_active_db_config()
#     chroma = get_chroma_client()
    
#     if not chroma:
#         return {
#             "graph_hits": [],
#             "chroma_ctx": [],
#             "fused_text_blocks": ["ChromaDB not available."]
#         }
    
#     # SEMANTIC SEARCH in ChromaDB
#     print(f"🔍 Semantic search in ChromaDB: '{query}'")
    
#     try:
#         chroma_docs = chroma.similarity_search(query, k=k_chroma)
#         print(f"📚 ChromaDB found {len(chroma_docs)} semantically similar papers")
        
#         # Fast-path: sort by year desc from metadata (kept)
#         chroma_docs = sorted(
#             chroma_docs,
#             key=lambda d: _safe_int((d.metadata or {}).get("year", 0)),
#             reverse=True
#         )
        
#     except Exception as e:
#         print(f"❌ ChromaDB search failed: {e}")
#         return {
#             "graph_hits": [],
#             "chroma_ctx": [],
#             "fused_text_blocks": ["ChromaDB search failed."]
#         }
    
#     if not chroma_docs:
#         return {
#             "graph_hits": [],
#             "chroma_ctx": [],
#             "fused_text_blocks": ["No semantically similar papers found."]
#         }
    
#     # Convert ChromaDB results to graph_hits format (for visualization only)
#     graph_hits = []
#     for doc in chroma_docs:
#         m = doc.metadata or {}
#         graph_hit = {**m}
#         graph_hit['paper_id'] = m.get('doi', '') or m.get('paper_id', '') or str(hash(doc.page_content))
#         graph_hit['title'] = m.get('title', 'Untitled')
#         graph_hit['year'] = _safe_int(m.get('year', 0))
#         graph_hit['doi'] = m.get('doi', '')
#         graph_hit['source'] = m.get('source', '')
#         graph_hit['researcher'] = m.get('researcher', 'Unknown')

#         # Extract authors
#         authors = []
#         for key, value in m.items():
#             if 'author' in key.lower():
#                 value_str = _safe_str(value)
#                 if value_str and value_str not in ['Unknown', 'N/A']:
#                     authors = re.split(r'[;,\|]', value_str)
#                     break
#         graph_hit['authors'] = [a.strip() for a in authors if a.strip()]
#         graph_hit['score'] = 0.0
#         graph_hits.append(graph_hit)
    
#     print(f"✅ Prepared {len(graph_hits)} papers for Neo4j graph visualization")
    
#     # Build display context
#     fused = []
#     meta = []
#     for idx, d in enumerate(chroma_docs, 1):
#         m = d.metadata or {}
#         year = _safe_int(m.get('year', 0))
#         year_str = f"({year})" if year > 0 else "(N/A)"
#         names = []
#         for key, value in m.items():
#             if any(term in key.lower() for term in ['author', 'researcher', 'name']):
#                 value_str = _safe_str(value)
#                 if value_str and value_str not in ['Unknown', 'N/A', '']:
#                     split_names = re.split(r'[;,\|]', value_str)
#                     names.extend([n.strip() for n in split_names if n.strip()])
#         names = list(set(names))
#         name_display = "; ".join(names[:5]) if names else "Unknown"
#         title = m.get('title', 'Untitled')
#         source = m.get('source', '')
#         doi = m.get('doi', '')
#         fused.append(
#             f"{idx}. {year_str} {title}\n"
#             f"   Authors: {name_display}\n"
#             f"   Source: {source} | DOI: {doi}"
#         )
#         meta.append(m)

#     fused_clean = _dedupe(fused)
#     chroma_texts = [d.page_content for d in chroma_docs]

#     # DB-layer sort (authoritative). This also preserves meta<->text alignment.
#     result = {
#         "graph_hits": graph_hits,
#         "chroma_ctx": chroma_texts,
#         "fused_text_blocks": fused_clean,
#         "fused_metadata": meta
#     }
#     result = sort_fused_result_newest_first(result)
#     return result

"""
hybrid_langchain_retriever.py - Pure semantic ChromaDB, Neo4j for graph only
(Updated Nov 2025)
- DB-level sort: apply newest → oldest using database_manager.sort_fused_result_newest_first
- Keeps year-desc sort of Chroma results as a fast-path; DB layer enforces final order
"""
import hashlib
import torch
import re
from typing import List
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from database_manager import get_active_db_config
# NEW: import the sorter from DB layer
from database_manager import sort_fused_result_newest_first

device = "cuda" if torch.cuda.is_available() else "cpu"
embed = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2", 
    model_kwargs={"device": device}
)

_chroma_clients = {}


def get_chroma_client():
    config = get_active_db_config()
    cache_key = f"{config.chroma_dir}_{config.chroma_collection}"
    
    if cache_key not in _chroma_clients:
        try:
            client = chromadb.PersistentClient(path=config.chroma_dir)
            _chroma_clients[cache_key] = Chroma(
                client=client,
                collection_name=config.chroma_collection,
                embedding_function=embed
            )
            print(f"✅ ChromaDB initialized: {config.chroma_collection}")
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            _chroma_clients[cache_key] = None
    
    return _chroma_clients[cache_key]


def clear_chroma_cache():
    global _chroma_clients
    _chroma_clients = {}


def _dedupe(lst):
    seen, out = set(), []
    for x in lst:
        h = hashlib.sha1(x.encode(errors="ignore")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(x)
    return out


def _safe_str(v):
    return str(v).strip() if v else ""


def _safe_int(v):
    try:
        return int(v) if v else 0
    except (ValueError, TypeError):
        return 0


def hybrid_search(query: str, k_chroma: int = 8, k_graph: int = 8):
    """
    Pure semantic search:
    1. ChromaDB semantic similarity (embeddings)
    2. Prepare Neo4j graph hits for visualization (no search logic here)
    3. DB-layer sort (newest → oldest) before returning fused blocks
    """
    config = get_active_db_config()
    chroma = get_chroma_client()
    
    if not chroma:
        return {
            "graph_hits": [],
            "chroma_ctx": [],
            "fused_text_blocks": ["ChromaDB not available."]
        }
    
    # SEMANTIC SEARCH in ChromaDB
    print(f"🔍 Semantic search in ChromaDB: '{query}'")
    
    try:
        chroma_docs = chroma.similarity_search(query, k=k_chroma)
        print(f"📚 ChromaDB found {len(chroma_docs)} semantically similar papers")
        
        # Fast-path: sort by year desc from metadata (kept)
        chroma_docs = sorted(
            chroma_docs,
            key=lambda d: _safe_int((d.metadata or {}).get("year", 0)),
            reverse=True
        )
        
    except Exception as e:
        print(f"❌ ChromaDB search failed: {e}")
        return {
            "graph_hits": [],
            "chroma_ctx": [],
            "fused_text_blocks": ["ChromaDB search failed."]
        }
    
    if not chroma_docs:
        return {
            "graph_hits": [],
            "chroma_ctx": [],
            "fused_text_blocks": ["No semantically similar papers found."]
        }
    
    # Convert ChromaDB results to graph_hits format (for visualization only)
    graph_hits = []
    for doc in chroma_docs:
        m = doc.metadata or {}
        graph_hit = {**m}
        graph_hit['paper_id'] = m.get('doi', '') or m.get('paper_id', '') or str(hash(doc.page_content))
        graph_hit['title'] = m.get('title', 'Untitled')
        graph_hit['year'] = _safe_int(m.get('year', 0))
        graph_hit['doi'] = m.get('doi', '')
        graph_hit['source'] = m.get('source', '')
        graph_hit['researcher'] = m.get('researcher', 'Unknown')

        # Extract authors
        authors = []
        for key, value in m.items():
            if 'author' in key.lower():
                value_str = _safe_str(value)
                if value_str and value_str not in ['Unknown', 'N/A']:
                    authors = re.split(r'[;,\|]', value_str)
                    break
        graph_hit['authors'] = [a.strip() for a in authors if a.strip()]
        graph_hit['score'] = 0.0
        graph_hits.append(graph_hit)
    
    print(f"✅ Prepared {len(graph_hits)} papers for Neo4j graph visualization")
    
    # Build display context
    fused = []
    meta = []
    for idx, d in enumerate(chroma_docs, 1):
        m = d.metadata or {}
        year = _safe_int(m.get('year', 0))
        year_str = f"({year})" if year > 0 else "(N/A)"
        names = []
        for key, value in m.items():
            if any(term in key.lower() for term in ['author', 'researcher', 'name']):
                value_str = _safe_str(value)
                if value_str and value_str not in ['Unknown', 'N/A', '']:
                    split_names = re.split(r'[;,\|]', value_str)
                    names.extend([n.strip() for n in split_names if n.strip()])
        names = list(set(names))
        name_display = "; ".join(names[:5]) if names else "Unknown"
        title = m.get('title', 'Untitled')
        source = m.get('source', '')
        doi = m.get('doi', '')
        fused.append(
            f"{idx}. {year_str} {title}\n"
            f"   Authors: {name_display}\n"
            f"   Source: {source} | DOI: {doi}"
        )
        meta.append(m)

    fused_clean = _dedupe(fused)
    chroma_texts = [d.page_content for d in chroma_docs]

    # DB-layer sort (authoritative). This also preserves meta<->text alignment.
    result = {
        "graph_hits": graph_hits,
        "chroma_ctx": chroma_texts,
        "fused_text_blocks": fused_clean,
        "fused_metadata": meta
    }
    result = sort_fused_result_newest_first(result)
    return result
