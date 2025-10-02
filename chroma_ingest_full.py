# # # # # import sqlite3, os, re
# # # # # import chromadb
# # # # # from tqdm import tqdm
# # # # # from chromadb.utils import embedding_functions
# # # # # import config_full as config

# # # # # DB_PATH = config.SQLITE_DB
# # # # # CHROMA_DIR = config.CHROMA_DIR
# # # # # BATCH_SIZE = 3000
# # # # # CHUNK_SIZE = 350  # ~300-400 chars, adjust for your use case

# # # # # chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
# # # # # local_embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # # # # metadata_coll = chroma_client.get_or_create_collection("paper_metadata", embedding_function=local_embed)
# # # # # summaries_coll = chroma_client.get_or_create_collection("paper_summaries", embedding_function=local_embed)
# # # # # fulltext_coll  = chroma_client.get_or_create_collection("paper_fulltext", embedding_function=local_embed)

# # # # # JUNK_RE = re.compile(r"^(readme|contents|fig\d+|about the author|writing center|journal)", re.I)
# # # # # DATE_RE = re.compile(r"(19|20)\d{2}")

# # # # # def clean_info(info: str):
# # # # #     if not info: return None, None
# # # # #     parts = info.split()
# # # # #     link = None
# # # # #     date = None
# # # # #     for p in parts:
# # # # #         if p.startswith("http"):
# # # # #             link = p
# # # # #         elif DATE_RE.match(p):
# # # # #             date = p
# # # # #     return link, date

# # # # # def chunk_text(text, size=CHUNK_SIZE):
# # # # #     # Basic chunker: splits by size, does not break in the middle of a word.
# # # # #     chunks = []
# # # # #     text = text.strip()
# # # # #     while len(text) > size:
# # # # #         split_at = text.rfind(" ", 0, size)
# # # # #         if split_at == -1:
# # # # #             split_at = size
# # # # #         chunks.append(text[:split_at].strip())
# # # # #         text = text[split_at:].strip()
# # # # #     if text: chunks.append(text)
# # # # #     return chunks

# # # # # def fetch_rows():
# # # # #     conn = sqlite3.connect(DB_PATH)
# # # # #     cur  = conn.cursor()
# # # # #     cur.execute("""
# # # # #         SELECT w.id, w.file_name, w.summary, w.full_text, ri.work_title, ri.authors, ri.researcher_name, ri.info
# # # # #         FROM works w
# # # # #         JOIN research_info ri ON w.id = ri.id
# # # # #         WHERE w.summary IS NOT NULL AND TRIM(w.summary) <> ''
# # # # #     """)
# # # # #     rows = cur.fetchall()
# # # # #     conn.close()
# # # # #     return rows

# # # # # def ingest_metadata(rows):
# # # # #     ids, docs, metas = [], [], []
# # # # #     for pid, fname, summary, fulltext, title, authors, researcher, info in rows:
# # # # #         if not title or JUNK_RE.match(title): continue
# # # # #         link, date = clean_info(info)
# # # # #         ids.append(f"meta-{pid}")
# # # # #         docs.append(title)
# # # # #         metas.append({
# # # # #             "authors": authors,
# # # # #             "researcher": researcher,
# # # # #             "file_name": fname,
# # # # #             "link": link,
# # # # #             "date": date
# # # # #         })
# # # # #         if len(ids) >= BATCH_SIZE:
# # # # #             metadata_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# # # # #             print(f"✅ Metadata batch ({len(ids)})")
# # # # #             ids, docs, metas = [], [], []
# # # # #     if ids:
# # # # #         metadata_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# # # # #         print(f"✅ Final metadata batch ({len(ids)})")

# # # # # def ingest_summaries(rows):
# # # # #     ids, docs, metas = [], [], []
# # # # #     for pid, fname, summary, fulltext, title, authors, researcher, info in rows:
# # # # #         if not summary or JUNK_RE.match(summary): continue
# # # # #         link, date = clean_info(info)
# # # # #         chunks = chunk_text(summary)
# # # # #         for i, chunk in enumerate(chunks):
# # # # #             ids.append(f"sum-{pid}-{i}")
# # # # #             docs.append(chunk)
# # # # #             metas.append({
# # # # #                 "title": title,
# # # # #                 "authors": authors,
# # # # #                 "researcher": researcher,
# # # # #                 "file_name": fname,
# # # # #                 "link": link,
# # # # #                 "date": date,
# # # # #                 "chunk_idx": i,
# # # # #                 "is_summary": True
# # # # #             })
# # # # #             if len(ids) >= BATCH_SIZE:
# # # # #                 summaries_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# # # # #                 print(f"✅ Summaries batch ({len(ids)})")
# # # # #                 ids, docs, metas = [], [], []
# # # # #     if ids:
# # # # #         summaries_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# # # # #         print(f"✅ Final summaries batch ({len(ids)})")

# # # # # def ingest_fulltext(rows):
# # # # #     ids, docs, metas = [], [], []
# # # # #     for pid, fname, summary, fulltext, title, authors, researcher, info in rows:
# # # # #         if not fulltext or len(fulltext) < 100: continue
# # # # #         link, date = clean_info(info)
# # # # #         chunks = chunk_text(fulltext)
# # # # #         for i, chunk in enumerate(chunks):
# # # # #             ids.append(f"full-{pid}-{i}")
# # # # #             docs.append(chunk)
# # # # #             metas.append({
# # # # #                 "title": title,
# # # # #                 "authors": authors,
# # # # #                 "researcher": researcher,
# # # # #                 "file_name": fname,
# # # # #                 "link": link,
# # # # #                 "date": date,
# # # # #                 "chunk_idx": i,
# # # # #                 "is_summary": False
# # # # #             })
# # # # #             if len(ids) >= BATCH_SIZE:
# # # # #                 fulltext_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# # # # #                 print(f"✅ Fulltext batch ({len(ids)})")
# # # # #                 ids, docs, metas = [], [], []
# # # # #     if ids:
# # # # #         fulltext_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# # # # #         print(f"✅ Final fulltext batch ({len(ids)})")

# # # # # if __name__ == "__main__":
# # # # #     rows = fetch_rows()
# # # # #     print(f"Found {len(rows)} rows for Chroma ingestion.")
# # # # #     ingest_metadata(rows)
# # # # #     ingest_summaries(rows)
# # # # #     ingest_fulltext(rows)
# # # # #     print("🎯 Chroma ingestion complete (all features, chunked)")


# # # # # chroma_ingest_full.py
# # # # import sqlite3, os, re
# # # # import chromadb
# # # # from tqdm import tqdm
# # # # from chromadb.utils import embedding_functions
# # # # import config_full as config

# # # # DB_PATH = config.SQLITE_DB
# # # # CHROMA_DIR = config.CHROMA_DIR
# # # # BATCH_SIZE = 3000
# # # # CHUNK_SIZE = 350  # ~300–400 chars
# # # # RESET_COLLECTIONS = True   # set False if you want to append

# # # # chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
# # # # local_embed = embedding_functions.SentenceTransformerEmbeddingFunction(
# # # #     model_name="sentence-transformers/all-MiniLM-L6-v2"
# # # # )

# # # # def reset_or_get_collection(name: str):
# # # #     if RESET_COLLECTIONS:
# # # #         try:
# # # #             chroma_client.delete_collection(name)
# # # #             print(f"🗑️  Deleted old collection '{name}'")
# # # #         except Exception:
# # # #             pass
# # # #     return chroma_client.get_or_create_collection(name, embedding_function=local_embed)

# # # # metadata_coll = reset_or_get_collection("paper_metadata")
# # # # summaries_coll = reset_or_get_collection("paper_summaries")
# # # # fulltext_coll  = reset_or_get_collection("paper_fulltext")

# # # # JUNK_RE = re.compile(r"^(readme|contents|fig\d+|about the author|writing center|journal)", re.I)
# # # # DATE_RE = re.compile(r"(19|20)\d{2}")

# # # # def clean_info(info: str):
# # # #     if not info: return None, None
# # # #     parts = info.split()
# # # #     link, date = None, None
# # # #     for p in parts:
# # # #         if p.startswith("http"):
# # # #             link = p
# # # #         elif DATE_RE.match(p):
# # # #             date = p
# # # #     return link, date

# # # # def chunk_text(text, size=CHUNK_SIZE):
# # # #     text = (text or "").strip()
# # # #     chunks = []
# # # #     while len(text) > size:
# # # #         split_at = text.rfind(" ", 0, size)
# # # #         if split_at == -1:
# # # #             split_at = size
# # # #         chunks.append(text[:split_at].strip())
# # # #         text = text[split_at:].strip()
# # # #     if text:
# # # #         chunks.append(text)
# # # #     return chunks

# # # # def fetch_rows():
# # # #     conn = sqlite3.connect(DB_PATH)
# # # #     cur  = conn.cursor()
# # # #     cur.execute("""
# # # #         SELECT w.id, w.file_name, w.summary, w.full_text,
# # # #                ri.work_title, ri.authors, ri.researcher_name, ri.info
# # # #         FROM works w
# # # #         LEFT JOIN research_info ri ON w.id = ri.id
# # # #         WHERE w.summary IS NOT NULL AND TRIM(w.summary) <> ''
# # # #     """)
# # # #     rows = cur.fetchall()
# # # #     conn.close()
# # # #     return rows

# # # # def batch_upsert(coll, ids, docs, metas, label="batch"):
# # # #     if not ids: return
# # # #     coll.upsert(ids=ids, documents=docs, metadatas=metas)
# # # #     print(f"✅ {label}: inserted {len(ids)} records")

# # # # def ingest_metadata(rows):
# # # #     ids, docs, metas = [], [], []
# # # #     for pid, fname, summary, fulltext, title, authors, researcher, info in tqdm(rows, desc="Metadata"):
# # # #         if not title or JUNK_RE.match(title): 
# # # #             continue
# # # #         link, date = clean_info(info or "")
# # # #         ids.append(f"meta-{pid}")
# # # #         docs.append(title)
# # # #         metas.append({
# # # #             "authors": authors,
# # # #             "researcher": researcher,
# # # #             "file_name": fname,
# # # #             "link": link,
# # # #             "date": date
# # # #         })
# # # #         if len(ids) >= BATCH_SIZE:
# # # #             batch_upsert(metadata_coll, ids, docs, metas, "Metadata batch")
# # # #             ids, docs, metas = [], [], []
# # # #     batch_upsert(metadata_coll, ids, docs, metas, "Final metadata batch")

# # # # def ingest_summaries(rows):
# # # #     ids, docs, metas = [], [], []
# # # #     for pid, fname, summary, fulltext, title, authors, researcher, info in tqdm(rows, desc="Summaries"):
# # # #         if not summary or JUNK_RE.match(summary): 
# # # #             continue
# # # #         link, date = clean_info(info or "")
# # # #         chunks = chunk_text(summary)
# # # #         for i, chunk in enumerate(chunks):
# # # #             ids.append(f"sum-{pid}-{i}")
# # # #             docs.append(chunk)
# # # #             metas.append({
# # # #                 "title": title,
# # # #                 "authors": authors,
# # # #                 "researcher": researcher,
# # # #                 "file_name": fname,
# # # #                 "link": link,
# # # #                 "date": date,
# # # #                 "chunk_idx": i,
# # # #                 "is_summary": True
# # # #             })
# # # #             if len(ids) >= BATCH_SIZE:
# # # #                 batch_upsert(summaries_coll, ids, docs, metas, "Summaries batch")
# # # #                 ids, docs, metas = [], [], []
# # # #     batch_upsert(summaries_coll, ids, docs, metas, "Final summaries batch")

# # # # def ingest_fulltext(rows):
# # # #     ids, docs, metas = [], [], []
# # # #     for pid, fname, summary, fulltext, title, authors, researcher, info in tqdm(rows, desc="Fulltext"):
# # # #         if not fulltext or len(fulltext) < 100: 
# # # #             continue
# # # #         link, date = clean_info(info or "")
# # # #         chunks = chunk_text(fulltext)
# # # #         for i, chunk in enumerate(chunks):
# # # #             ids.append(f"full-{pid}-{i}")
# # # #             docs.append(chunk)
# # # #             metas.append({
# # # #                 "title": title,
# # # #                 "authors": authors,
# # # #                 "researcher": researcher,
# # # #                 "file_name": fname,
# # # #                 "link": link,
# # # #                 "date": date,
# # # #                 "chunk_idx": i,
# # # #                 "is_summary": False
# # # #             })
# # # #             if len(ids) >= BATCH_SIZE:
# # # #                 batch_upsert(fulltext_coll, ids, docs, metas, "Fulltext batch")
# # # #                 ids, docs, metas = [], [], []
# # # #     batch_upsert(fulltext_coll, ids, docs, metas, "Final fulltext batch")

# # # # if __name__ == "__main__":
# # # #     rows = fetch_rows()
# # # #     print(f"📦 Found {len(rows)} rows for ingestion from {DB_PATH}")
# # # #     ingest_metadata(rows)
# # # #     ingest_summaries(rows)
# # # #     ingest_fulltext(rows)
# # # #     print("🎯 Chroma ingestion complete (metadata, summaries, fulltext)")


# # # # chroma_ingest_full_resume.py
# # # import sqlite3, re
# # # import chromadb
# # # from tqdm import tqdm
# # # from chromadb.utils import embedding_functions
# # # import config_full as config

# # # DB_PATH = config.SQLITE_DB
# # # CHROMA_DIR = config.CHROMA_DIR
# # # BATCH_SIZE = 3000
# # # CHUNK_SIZE = 350

# # # # Init Chroma client + embedding
# # # chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
# # # local_embed = embedding_functions.SentenceTransformerEmbeddingFunction(
# # #     model_name="sentence-transformers/all-MiniLM-L6-v2"
# # # )

# # # metadata_coll = chroma_client.get_or_create_collection("paper_metadata", embedding_function=local_embed)
# # # summaries_coll = chroma_client.get_or_create_collection("paper_summaries", embedding_function=local_embed)
# # # fulltext_coll  = chroma_client.get_or_create_collection("paper_fulltext", embedding_function=local_embed)

# # # JUNK_RE = re.compile(r"^(readme|contents|fig\d+|about the author|writing center|journal)", re.I)
# # # DATE_RE = re.compile(r"(19|20)\d{2}")

# # # def clean_info(info: str):
# # #     if not info: return None, None
# # #     parts = info.split()
# # #     link, date = None, None
# # #     for p in parts:
# # #         if p.startswith("http"):
# # #             link = p
# # #         elif DATE_RE.match(p):
# # #             date = p
# # #     return link, date

# # # def chunk_text(text, size=CHUNK_SIZE):
# # #     chunks, text = [], text.strip()
# # #     while len(text) > size:
# # #         split_at = text.rfind(" ", 0, size)
# # #         if split_at == -1: split_at = size
# # #         chunks.append(text[:split_at].strip())
# # #         text = text[split_at:].strip()
# # #     if text: chunks.append(text)
# # #     return chunks

# # # def fetch_rows():
# # #     conn = sqlite3.connect(DB_PATH)
# # #     cur  = conn.cursor()
# # #     cur.execute("""
# # #         SELECT w.id, w.file_name, w.summary, w.full_text, 
# # #                ri.work_title, ri.authors, ri.researcher_name, ri.info
# # #         FROM works w
# # #         JOIN research_info ri ON w.id = ri.id
# # #         WHERE w.summary IS NOT NULL AND TRIM(w.summary) <> ''
# # #     """)
# # #     rows = cur.fetchall()
# # #     conn.close()
# # #     return rows

# # # def already_ingested_ids(collection):
# # #     """Return set of existing IDs in a Chroma collection."""
# # #     existing = set()
# # #     try:
# # #         offset, batch_size = 0, 5000
# # #         while True:
# # #             result = collection.get(ids=None, limit=batch_size, offset=offset)
# # #             if not result or "ids" not in result or not result["ids"]:
# # #                 break
# # #             existing.update(result["ids"])
# # #             offset += batch_size
# # #     except Exception:
# # #         pass
# # #     return existing

# # # def ingest_metadata(rows):
# # #     existing = already_ingested_ids(metadata_coll)
# # #     ids, docs, metas = [], [], []
# # #     for pid, fname, summary, fulltext, title, authors, researcher, info in tqdm(rows, desc="Metadata"):
# # #         if not title or JUNK_RE.match(title): continue
# # #         doc_id = f"meta-{pid}"
# # #         if doc_id in existing: continue
# # #         link, date = clean_info(info)
# # #         ids.append(doc_id)
# # #         docs.append(title)
# # #         metas.append({
# # #             "authors": authors, "researcher": researcher, "file_name": fname,
# # #             "link": link, "date": date
# # #         })
# # #         if len(ids) >= BATCH_SIZE:
# # #             metadata_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# # #             print(f"✅ Metadata batch ({len(ids)})")
# # #             ids, docs, metas = [], [], []
# # #     if ids:
# # #         metadata_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# # #         print(f"✅ Final metadata batch ({len(ids)})")

# # # def ingest_summaries(rows):
# # #     existing = already_ingested_ids(summaries_coll)
# # #     ids, docs, metas = [], [], []
# # #     for pid, fname, summary, fulltext, title, authors, researcher, info in tqdm(rows, desc="Summaries"):
# # #         if not summary or JUNK_RE.match(summary): continue
# # #         chunks = chunk_text(summary)
# # #         for i, chunk in enumerate(chunks):
# # #             doc_id = f"sum-{pid}-{i}"
# # #             if doc_id in existing: continue
# # #             link, date = clean_info(info)
# # #             ids.append(doc_id)
# # #             docs.append(chunk)
# # #             metas.append({
# # #                 "title": title, "authors": authors, "researcher": researcher,
# # #                 "file_name": fname, "link": link, "date": date,
# # #                 "chunk_idx": i, "is_summary": True
# # #             })
# # #             if len(ids) >= BATCH_SIZE:
# # #                 summaries_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# # #                 print(f"✅ Summaries batch ({len(ids)})")
# # #                 ids, docs, metas = [], [], []
# # #     if ids:
# # #         summaries_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# # #         print(f"✅ Final summaries batch ({len(ids)})")

# # # def ingest_fulltext(rows):
# # #     existing = already_ingested_ids(fulltext_coll)
# # #     ids, docs, metas = [], [], []
# # #     for pid, fname, summary, fulltext, title, authors, researcher, info in tqdm(rows, desc="Fulltext"):
# # #         if not fulltext or len(fulltext) < 100: continue
# # #         chunks = chunk_text(fulltext)
# # #         for i, chunk in enumerate(chunks):
# # #             doc_id = f"full-{pid}-{i}"
# # #             if doc_id in existing: continue
# # #             link, date = clean_info(info)
# # #             ids.append(doc_id)
# # #             docs.append(chunk)
# # #             metas.append({
# # #                 "title": title, "authors": authors, "researcher": researcher,
# # #                 "file_name": fname, "link": link, "date": date,
# # #                 "chunk_idx": i, "is_summary": False
# # #             })
# # #             if len(ids) >= BATCH_SIZE:
# # #                 fulltext_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# # #                 print(f"✅ Fulltext batch ({len(ids)})")
# # #                 ids, docs, metas = [], [], []
# # #     if ids:
# # #         fulltext_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# # #         print(f"✅ Final fulltext batch ({len(ids)})")

# # # if __name__ == "__main__":
# # #     rows = fetch_rows()
# # #     print(f"Found {len(rows)} rows for Chroma ingestion.")
# # #     ingest_metadata(rows)
# # #     ingest_summaries(rows)
# # #     ingest_fulltext(rows)
# # #     print("🎯 Chroma ingestion complete with resume support")


# # # chroma_ingest.py
# # import sqlite3, re
# # import chromadb
# # from tqdm import tqdm
# # from chromadb.utils import embedding_functions
# # import config_full as config

# # DB_PATH = config.SQLITE_DB
# # CHROMA_DIR = config.CHROMA_DIR
# # BATCH_SIZE = 3000
# # CHUNK_SIZE = 350

# # # Init Chroma client + embedding
# # chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
# # local_embed = embedding_functions.SentenceTransformerEmbeddingFunction(
# #     model_name="sentence-transformers/all-MiniLM-L6-v2"
# # )

# # metadata_coll = chroma_client.get_or_create_collection(
# #     "paper_metadata", embedding_function=local_embed
# # )
# # summaries_coll = chroma_client.get_or_create_collection(
# #     "paper_summaries", embedding_function=local_embed
# # )
# # fulltext_coll  = chroma_client.get_or_create_collection(
# #     "paper_fulltext", embedding_function=local_embed
# # )

# # JUNK_RE = re.compile(r"^(readme|contents|fig\d+|about the author|writing center|journal)", re.I)
# # DATE_RE = re.compile(r"(?:19|20)\d{2}")

# # def clean_info(info: str):
# #     if not info: return None, None
# #     parts = info.split()
# #     link, date = None, None
# #     for p in parts:
# #         if p.startswith("http"):
# #             link = p
# #         elif DATE_RE.match(p):
# #             date = p
# #     return link, date

# # def chunk_text(text, size=CHUNK_SIZE):
# #     chunks, text = [], (text or "").strip()
# #     while len(text) > size:
# #         split_at = text.rfind(" ", 0, size)
# #         if split_at == -1: split_at = size
# #         chunks.append(text[:split_at].strip())
# #         text = text[split_at:].strip()
# #     if text: chunks.append(text)
# #     return chunks

# # def fetch_rows():
# #     conn = sqlite3.connect(DB_PATH)
# #     cur  = conn.cursor()
# #     cur.execute("""
# #         SELECT w.id, w.file_name, w.summary, w.full_text, 
# #                ri.work_title, ri.authors, ri.researcher_name, ri.info
# #         FROM works w
# #         JOIN research_info ri ON w.id = ri.id
# #         WHERE w.summary IS NOT NULL AND TRIM(w.summary) <> ''
# #     """)
# #     rows = cur.fetchall()
# #     conn.close()
# #     return rows

# # def already_ingested_ids(collection):
# #     """Return set of existing IDs in a Chroma collection (paged)."""
# #     existing = set()
# #     try:
# #         offset, batch_size = 0, 5000
# #         while True:
# #             result = collection.get(
# #                 ids=None, where={}, limit=batch_size, offset=offset, include=["ids"]
# #             )
# #             ids = result.get("ids") if isinstance(result, dict) else None
# #             if not ids:
# #                 break
# #             existing.update(ids)
# #             if len(ids) < batch_size:
# #                 break
# #             offset += batch_size
# #     except Exception as e:
# #         print("Warning: could not list existing IDs:", e)
# #     return existing

# # def ingest_metadata(rows):
# #     existing = already_ingested_ids(metadata_coll)
# #     ids, docs, metas = [], [], []
# #     for pid, fname, summary, fulltext, title, authors, researcher, info in tqdm(rows, desc="Metadata"):
# #         if not title or JUNK_RE.match(title): continue
# #         doc_id = f"meta-{pid}"
# #         if doc_id in existing: continue
# #         link, date = clean_info(info)
# #         ids.append(doc_id)
# #         docs.append(title)
# #         metas.append({
# #             "authors": authors, "researcher": researcher, "file_name": fname,
# #             "link": link, "date": date
# #         })
# #         if len(ids) >= BATCH_SIZE:
# #             metadata_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# #             print(f"Metadata batch ({len(ids)})")
# #             ids, docs, metas = [], [], []
# #     if ids:
# #         metadata_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# #         print(f"Final metadata batch ({len(ids)})")

# # def ingest_summaries(rows):
# #     existing = already_ingested_ids(summaries_coll)
# #     ids, docs, metas = [], [], []
# #     for pid, fname, summary, fulltext, title, authors, researcher, info in tqdm(rows, desc="Summaries"):
# #         if not summary or JUNK_RE.match(summary): continue
# #         chunks = chunk_text(summary)
# #         link, date = clean_info(info)
# #         for i, chunk in enumerate(chunks):
# #             doc_id = f"sum-{pid}-{i}"
# #             if doc_id in existing: continue
# #             ids.append(doc_id)
# #             docs.append(chunk)
# #             metas.append({
# #                 "title": title, "authors": authors, "researcher": researcher,
# #                 "file_name": fname, "link": link, "date": date,
# #                 "chunk_idx": i, "is_summary": True
# #             })
# #             if len(ids) >= BATCH_SIZE:
# #                 summaries_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# #                 print(f"Summaries batch ({len(ids)})")
# #                 ids, docs, metas = [], [], []
# #     if ids:
# #         summaries_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# #         print(f"Final summaries batch ({len(ids)})")

# # def ingest_fulltext(rows):
# #     existing = already_ingested_ids(fulltext_coll)
# #     ids, docs, metas = [], [], []
# #     for pid, fname, summary, fulltext, title, authors, researcher, info in tqdm(rows, desc="Fulltext"):
# #         if not fulltext or len(fulltext) < 100: continue
# #         chunks = chunk_text(fulltext)
# #         link, date = clean_info(info)
# #         for i, chunk in enumerate(chunks):
# #             doc_id = f"full-{pid}-{i}"
# #             if doc_id in existing: continue
# #             ids.append(doc_id)
# #             docs.append(chunk)
# #             metas.append({
# #                 "title": title, "authors": authors, "researcher": researcher,
# #                 "file_name": fname, "link": link, "date": date,
# #                 "chunk_idx": i, "is_summary": False
# #             })
# #             if len(ids) >= BATCH_SIZE:
# #                 fulltext_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# #                 print(f"Fulltext batch ({len(ids)})")
# #                 ids, docs, metas = [], [], []
# #     if ids:
# #         fulltext_coll.upsert(ids=ids, documents=docs, metadatas=metas)
# #         print(f"Final fulltext batch ({len(ids)})")

# # if __name__ == "__main__":
# #     rows = fetch_rows()
# #     print(f"Found {len(rows)} rows for Chroma ingestion.")
# #     ingest_metadata(rows)
# #     ingest_summaries(rows)
# #     ingest_fulltext(rows)
# #     print("Chroma ingestion complete with resume support")


# # chroma_ingest_full.py
# """
# Ingest SQLite content into Chroma (metadata, summaries, fulltext) with resume support.

# Collections:
#   - paper_metadata : title-level docs (1 per paper)
#   - paper_summaries: chunked summary text (if available)
#   - paper_fulltext : chunked full text (if available)

# Config:
#   - Uses paths and knobs from config_full.py
#   - Honors:
#       SQLITE_DB, CHROMA_DIR, SENTENCE_TFORMER,
#       CHROMA_BATCH, CHUNK_SIZE (for text chunking)

# Notes:
#   - Idempotent: checks for existing IDs and only adds missing ones.
#   - Robust chunking.
#   - Avoids hard-coding; all paths come from config_full.py (env overridable).
# """

# import sqlite3
# import re
# from typing import Iterable, List, Tuple
# from tqdm import tqdm

# import chromadb
# from chromadb.utils import embedding_functions

# import config_full as config

# # --------------------------- constants / regex --------------------------------

# JUNK_RE = re.compile(r"^(readme|contents|fig\d+|about the author|writing center|journal)", re.I)
# DATE_RE = re.compile(r"(19|20)\d{2}")

# CHUNK_SIZE = int(getattr(config, "CHUNK_SIZE", 1000))  # safe default
# CHROMA_BATCH = int(getattr(config, "CHROMA_BATCH", 3000))

# # --------------------------- Chroma client / embeddings -----------------------

# client = chromadb.PersistentClient(path=config.CHROMA_DIR)

# # Use SentenceTransformerEmbeddingFunction (works on CPU/GPU depending on env/torch)
# # We don't force dtype to avoid incompatibilities across versions.
# local_embed = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name=config.SENTENCE_TFORMER
# )

# # Create/get collections (embedding_function bound here for server-side embeddings)
# coll_meta = client.get_or_create_collection(
#     "paper_metadata", embedding_function=local_embed
# )
# coll_sum = client.get_or_create_collection(
#     "paper_summaries", embedding_function=local_embed
# )
# coll_full = client.get_or_create_collection(
#     "paper_fulltext", embedding_function=local_embed
# )

# # --------------------------- helpers -----------------------------------------

# def clean_info(info: str):
#     """Extract first http(s) link and the first year-like substring."""
#     if not info:
#         return None, None
#     parts = info.split()
#     link, date = None, None
#     for p in parts:
#         if p.startswith("http"):
#             link = p
#         elif DATE_RE.match(p):
#             date = p
#     return link, date

# def chunk_text(text: str, size: int = CHUNK_SIZE) -> List[str]:
#     text = (text or "").strip()
#     if not text:
#         return []
#     chunks: List[str] = []
#     while len(text) > size:
#         split_at = text.rfind(" ", 0, size)
#         if split_at == -1:
#             split_at = size
#         chunk = text[:split_at].strip()
#         if chunk:
#             chunks.append(chunk)
#         text = text[split_at:].strip()
#     if text:
#         chunks.append(text)
#     return chunks

# def fetch_rows() -> List[Tuple]:
#     conn = sqlite3.connect(config.SQLITE_DB)
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT w.id, w.file_name, w.summary, w.full_text, 
#                ri.work_title, ri.authors, ri.researcher_name, ri.info
#         FROM works w
#         JOIN research_info ri ON w.id = ri.id
#         WHERE w.summary IS NOT NULL AND TRIM(w.summary) <> ''
#            OR (w.full_text IS NOT NULL AND TRIM(w.full_text) <> '')
#            OR (ri.work_title IS NOT NULL AND TRIM(ri.work_title) <> '')
#     """)
#     rows = cur.fetchall()
#     conn.close()
#     return rows

# def already_ingested_ids(collection) -> set:
#     """
#     Collect existing IDs in a Chroma collection by paging through results.
#     """
#     existing = set()
#     try:
#         offset, batch = 0, 5000
#         while True:
#             res = collection.get(ids=None, limit=batch, offset=offset)
#             ids = res.get("ids") if isinstance(res, dict) else None
#             if not ids:
#                 break
#             existing.update(ids)
#             if len(ids) < batch:
#                 break
#             offset += batch
#     except Exception:
#         # If anything goes wrong, just treat as empty; ingestion will dedupe on upsert keys.
#         pass
#     return existing

# def upsert_batched(collection, ids: List[str], docs: List[str], metas: List[dict], batch_size: int = CHROMA_BATCH, label: str = ""):
#     count = 0
#     for i in range(0, len(ids), batch_size):
#         batch_ids = ids[i:i+batch_size]
#         batch_docs = docs[i:i+batch_size]
#         batch_metas = metas[i:i+batch_size]
#         collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
#         count += len(batch_ids)
#         if label:
#             print(f"✅ {label} batch upserted: {len(batch_ids)} (total {count})")
#     return count

# # --------------------------- ingestion routines -------------------------------

# def ingest_metadata(rows: Iterable[Tuple]):
#     existing = already_ingested_ids(coll_meta)
#     ids, docs, metas = [], [], []

#     for pid, fname, summary, fulltext, title, authors, researcher, info in tqdm(rows, desc="Metadata"):
#         title = (title or "").strip()
#         if not title or JUNK_RE.match(title):
#             continue
#         doc_id = f"meta-{pid}"
#         if doc_id in existing:
#             continue
#         link, date = clean_info(info or "")
#         ids.append(doc_id)
#         docs.append(title)
#         metas.append({
#             "authors": authors or "",
#             "researcher": researcher or "",
#             "file_name": fname or "",
#             "link": link,
#             "date": date
#         })

#     if ids:
#         upsert_batched(coll_meta, ids, docs, metas, label="Metadata")
#     print(f"📚 Metadata done: +{len(ids)} new")

# def ingest_summaries(rows: Iterable[Tuple]):
#     existing = already_ingested_ids(coll_sum)
#     ids, docs, metas = [], [], []

#     for pid, fname, summary, fulltext, title, authors, researcher, info in tqdm(rows, desc="Summaries"):
#         summary = (summary or "").strip()
#         if not summary or JUNK_RE.match(summary):
#             continue
#         chunks = chunk_text(summary)
#         if not chunks:
#             continue
#         link, date = clean_info(info or "")
#         for i, chunk in enumerate(chunks):
#             doc_id = f"sum-{pid}-{i}"
#             if doc_id in existing:
#                 continue
#             ids.append(doc_id)
#             docs.append(chunk)
#             metas.append({
#                 "title": title or "",
#                 "authors": authors or "",
#                 "researcher": researcher or "",
#                 "file_name": fname or "",
#                 "link": link,
#                 "date": date,
#                 "chunk_idx": i,
#                 "is_summary": True
#             })

#     if ids:
#         upsert_batched(coll_sum, ids, docs, metas, label="Summaries")
#     print(f"📝 Summaries done: +{len(ids)} new")

# def ingest_fulltext(rows: Iterable[Tuple]):
#     existing = already_ingested_ids(coll_full)
#     ids, docs, metas = [], [], []

#     for pid, fname, summary, fulltext, title, authors, researcher, info in tqdm(rows, desc="Fulltext"):
#         fulltext = (fulltext or "").strip()
#         if len(fulltext) < 100:
#             continue
#         chunks = chunk_text(fulltext)
#         if not chunks:
#             continue
#         link, date = clean_info(info or "")
#         for i, chunk in enumerate(chunks):
#             doc_id = f"full-{pid}-{i}"
#             if doc_id in existing:
#                 continue
#             ids.append(doc_id)
#             docs.append(chunk)
#             metas.append({
#                 "title": title or "",
#                 "authors": authors or "",
#                 "researcher": researcher or "",
#                 "file_name": fname or "",
#                 "link": link,
#                 "date": date,
#                 "chunk_idx": i,
#                 "is_summary": False
#             })

#     if ids:
#         upsert_batched(coll_full, ids, docs, metas, label="Fulltext")
#     print(f"📄 Fulltext done: +{len(ids)} new")

# # --------------------------- main --------------------------------------------

# if __name__ == "__main__":
#     print("— Chroma Ingest (resume-friendly) —")
#     print(f"SQLite    : {config.SQLITE_DB}")
#     print(f"Chroma dir: {config.CHROMA_DIR}")
#     print(f"Embedder  : {config.SENTENCE_TFORMER}")
#     print(f"Chunk size: {CHUNK_SIZE} | Batch: {CHROMA_BATCH}")

#     rows = fetch_rows()
#     print(f"Found {len(rows)} candidate rows.")

#     if not rows:
#         print("Nothing to ingest. Check your SQLite tables (works, research_info).")
#     else:
#         ingest_metadata(rows)
#         ingest_summaries(rows)
#         ingest_fulltext(rows)
#         print("🎯 Chroma ingestion complete (with resume support).")

import os
import sqlite3
import time
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import config_full as config

DB_PATH = config.DB_PATH
CHROMA_DIR = config.CHROMA_DIR
BATCH_SIZE = 500   # 🔥 tune for speed/memory (500–1000 works well on 32GB RAM)

# -------- Helpers --------
def split_into_halves(text: str):
    """Split into 2 halves like HW4, to keep context large and coherent."""
    if not text:
        return []
    mid = len(text) // 2
    return [text[:mid], text[mid:]]

def build_doc(row):
    rid, researcher, title, authors, info, doi, pub_date, file_name, summary, fulltext = row
    return (
        rid,
        (
            f"Researcher: {researcher or 'Unknown'}\n"
            f"Title: {title or 'Untitled'}\n"
            f"Authors: {authors or 'N/A'}\n"
            f"Info: {info or 'N/A'}\n"
            f"DOI: {doi or 'N/A'}\n"
            f"Publication Date: {pub_date or 'N/A'}\n"
            f"File: {file_name or 'N/A'}\n\n"
            f"Summary: {summary or 'N/A'}\n\n"
            f"Fulltext:\n{fulltext or 'N/A'}"
        )
    )

# -------- Main Ingestion --------
def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # ✅ Local embeddings (no API key, GPU auto-detected)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/e5-base-v2"   # strong open-source sentence embedder
    )

    collection = client.get_or_create_collection(
        name="papers_all",
        embedding_function=embedder
    )

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ✅ Merge research_info with works
    cur.execute("""
        SELECT r.id, r.researcher_name, r.work_title, r.authors, r.info, r.doi, r.publication_date,
               w.file_name, w.summary, w.full_text
        FROM research_info r
        LEFT JOIN works w ON r.id = w.id
    """)
    rows = cur.fetchall()

    print(f"🔎 Found {len(rows)} combined rows to ingest.")

    total_start = time.time()

    # Process in batches
    for start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Ingesting", unit="batch"):
        batch = rows[start:start + BATCH_SIZE]

        docs, ids, metas = [], [], []

        for row in batch:
            rid, text = build_doc(row)
            chunks = split_into_halves(text)

            for i, chunk in enumerate(chunks, start=1):
                docs.append(chunk)
                ids.append(f"{rid}_part{i}")
                metas.append({
                    "id": rid,
                    "researcher": row[1] or "Unknown",
                    "title": row[2] or "Untitled",
                    "authors": row[3] or "N/A",
                    "doi": row[5] or "N/A",
                    "publication_date": row[6] or "N/A",
                    "file_name": row[7] or "N/A",
                    "chunk": i,
                })

        # ✅ Bulk insert with throughput logging
        batch_start = time.time()
        collection.add(documents=docs, ids=ids, metadatas=metas)
        batch_end = time.time()

        speed = len(docs) / (batch_end - batch_start + 1e-6)
        print(f"⚡ Batch {start//BATCH_SIZE+1}: {len(docs)} docs at {speed:.2f} docs/sec")

    total_end = time.time()
    print(f"✅ Ingestion complete — {len(rows)} works stored in `papers_all` collection.")
    print(f"⏱️ Total time: {total_end - total_start:.2f} seconds")

if __name__ == "__main__":
    main()
