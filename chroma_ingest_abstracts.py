"""
chroma_ingest_abstracts.py
Builds a ChromaDB collection 'abstracts_all' from abstracts_only.db.
Each abstract becomes a vector document with full metadata:
  doi, title, source, year, researcher_name, authors, info, publication_date
Includes safety checks for database existence and permissions.
"""

import os
import sqlite3
import time
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import config_full as config
import sys

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = config.DB_PATH
CHROMA_DIR = config.CHROMA_DIR
COLLECTION_NAME = "abstracts_all"
BATCH_SIZE = 500

# ─────────────────────────────────────────────────────────────────────────────
# SAFETY + DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
def validate_db_path(path):
    """Check if the SQLite database exists and is readable."""
    print(f"🔍 Checking database path: {path}")

    # Folder existence
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ Folder not found: {folder}")

    # DB existence
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Database file not found: {path}")

    # Permissions
    if not os.access(path, os.R_OK):
        raise PermissionError(f"❌ No read permission for: {path}")

    if not os.access(path, os.W_OK):
        print(f"⚠️ Warning: No write permission for: {path} (read-only mode).")

    # Try connecting
    try:
        conn = sqlite3.connect(path)
        conn.execute("SELECT name FROM sqlite_master LIMIT 1;")
        conn.close()
        print(f"✅ Database connection test passed: {path}")
    except sqlite3.Error as e:
        raise sqlite3.OperationalError(f"❌ SQLite cannot open file: {path}\nError: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def safe_str(v):
    if v is None:
        return ""
    return str(v).strip()


def check_columns(cur):
    """Ensure all expected columns exist in abstracts_only."""
    cur.execute("PRAGMA table_info(abstracts_only);")
    cols = {r[1] for r in cur.fetchall()}
    required = {
        "doi", "title", "abstract", "source", "year",
        "researcher_name", "work_title", "authors", "info", "publication_date"
    }
    missing = required - cols
    if missing:
        raise ValueError(f"❌ Missing columns in abstracts_only.db: {missing}")
    print("✅ All expected columns present in abstracts_only.db.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INGEST
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("────────────────────────────────────────────────────────")
    print("🚀 Chroma Abstracts Ingest Starting")
    print(f"📂 Using database: {DB_PATH}")
    print(f"🧠 Chroma store dir: {CHROMA_DIR}")
    print("────────────────────────────────────────────────────────")

    # Validate database access before continuing
    try:
        validate_db_path(DB_PATH)
    except Exception as e:
        print(e)
        sys.exit(1)

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/e5-base-v2"
    )

    # Recreate collection cleanly
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️ Old '{COLLECTION_NAME}' collection removed.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder
    )

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    check_columns(cur)

    # Fetch rows with abstracts
    cur.execute("""
        SELECT doi, title, abstract, source, year,
               researcher_name, work_title, authors, info, publication_date
        FROM abstracts_only
        WHERE abstract IS NOT NULL AND abstract != ''
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("⚠️ No abstracts found in database. Exiting.")
        return

    print(f"📚 Found {len(rows)} enriched abstracts to ingest into Chroma.")
    start_all = time.time()

    # Batch ingestion
    for start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Ingesting", unit="batch"):
        batch = rows[start:start+BATCH_SIZE]
        ids, docs, metas = [], [], []

        for idx, (doi, title, abstract, source, year,
                  researcher_name, work_title, authors, info, pub_date) in enumerate(batch):
            if not abstract:
                continue

            ids.append(f"{start}_{idx}")
            docs.append(safe_str(abstract))
            metas.append({
                "doi": safe_str(doi),
                "title": safe_str(title),
                "source": safe_str(source),
                "year": safe_str(year),
                "researcher": safe_str(researcher_name),
                "work_title": safe_str(work_title),
                "authors": safe_str(authors),
                "info": safe_str(info),
                "publication_date": safe_str(pub_date),
            })

        if docs:
            t0 = time.time()
            collection.add(ids=ids, documents=docs, metadatas=metas)
            speed = len(docs) / (time.time() - t0 + 1e-6)
            print(f"⚡ Batch {start//BATCH_SIZE+1}: {len(docs)} docs at {speed:.2f} docs/sec")

    print(f"✅ Ingest complete — {len(rows)} abstracts stored in '{COLLECTION_NAME}'.")
    print(f"⏱️ Total time: {time.time() - start_all:.2f}s")
    print("────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
 