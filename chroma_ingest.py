import os
import sqlite3
import time
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import config_full as config

DB_PATH = config.SQLITE_DB          # ← fix
CHROMA_DIR = config.CHROMA_DIR
BATCH_SIZE = 500

def split_into_halves(text: str):
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

def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/e5-base-v2"
    )

    collection = client.get_or_create_collection(
        name="papers_all",
        embedding_function=embedder
    )

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT r.id, r.researcher_name, r.work_title, r.authors, r.info, r.doi, r.publication_date,
               w.file_name, w.summary, w.full_text
        FROM research_info r
        LEFT JOIN works w ON r.id = w.id
    """)
    rows = cur.fetchall()
    conn.close()

    print(f"🔎 Found {len(rows)} combined rows to ingest.")

    total_start = time.time()
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

        batch_start = time.time()
        collection.add(documents=docs, ids=ids, metadatas=metas)
        speed = len(docs) / (time.time() - batch_start + 1e-6)
        print(f"⚡ Batch {start//BATCH_SIZE+1}: {len(docs)} docs at {speed:.2f} docs/sec")

    print(f"✅ Ingestion complete — {len(rows)} works stored in `papers_all`.")
    print(f"⏱️ Total time: {time.time() - total_start:.2f} seconds")

if __name__ == "__main__":
    main()
