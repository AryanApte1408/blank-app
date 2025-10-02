# # # config_full.py
# # import os

# # # --- SQLite (original full DB) ---
# # SQLITE_DB = r"D:\OSPO\KG-RAG1\researchers_fixed.db"   # your original full DB

# # # --- Neo4j ---
# # NEO4J_URI  = "bolt://localhost:7687"
# # NEO4J_USER = "neo4j"
# # NEO4J_PASS = "OSPOlol@1234"
# # NEO4J_DB   = "syr-rag"   # will be created/used

# # # --- Chroma ---
# # CHROMA_DIR = r"D:\OSPO\KG-RAG1\chroma_store_full"

# # # --- LLaMA (Transformers format dir; not GGUF) ---
# # LLAMA_MODEL_PATH = r"D:\OSPO\KG-RAG1\Llama-3.2-1B-Instruct"

# # # --- Ingest controls ---
# # CHUNK_SIZE   = 1000
# # CHUNK_OVERLAP= 200
# # BATCH_SIZE   = 2000
# # CHROMA_BATCH = 3000
# # PARALLEL_JOBS= 4

# # # --- Embeddings ---
# # SENTENCE_TFORMER = "sentence-transformers/all-MiniLM-L6-v2"


# """
# config_full.py
# Centralized, no-hardcode configuration for KG+Chroma RAG (Neo4j Community Edition friendly).

# Override anything via environment variables. Sensible defaults are used otherwise.

# Key env vars (all optional):
# - PROJECT_ROOT
# - SQLITE_DB
# - CHROMA_DIR
# - HF_CACHE_DIR
# - NEO4J_URI
# - NEO4J_USER
# - NEO4J_PASS
# - NEO4J_DB                # ignored on Community Edition; kept for compatibility
# - SENTENCE_TFORMER
# - LLM_MODEL_PATH          # local HF path or model id; used by chat_app/rag_pipeline if needed
# - DEVICE                  # "cuda", "cpu", or "auto"
# - CUDA_VISIBLE_DEVICES    # standard CUDA selection; not read here, but respected by libs
# - CHUNK_SIZE
# - CHUNK_OVERLAP
# - BATCH_SIZE
# - CHROMA_BATCH
# - PARALLEL_JOBS
# """

# import os
# from pathlib import Path

# # ---- paths -------------------------------------------------------------------

# def _default_root() -> Path:
#     # If PROJECT_ROOT not set, use the directory that contains this file
#     try:
#         return Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
#     except Exception:
#         return Path.cwd()

# PROJECT_ROOT = _default_root()

# def _resolve_dir(var_name: str, default_rel: str) -> Path:
#     p = os.environ.get(var_name, str(PROJECT_ROOT / default_rel))
#     path = Path(p)
#     path.mkdir(parents=True, exist_ok=True)
#     return path

# def _resolve_file(var_name: str, default_rel: str) -> Path:
#     p = os.environ.get(var_name, str(PROJECT_ROOT / default_rel))
#     return Path(p)

# # SQLite DB file (your source catalog: works, research_info, etc.)
# SQLITE_DB: str = str(_resolve_file("SQLITE_DB", "researchers_fixed.db"))

# # Chroma persistent store directory
# CHROMA_DIR: str = str(_resolve_dir("CHROMA_DIR", "chroma_store"))

# # HuggingFace cache dir (keeps models off the network after first pull)
# HF_CACHE_DIR: str = str(_resolve_dir("HF_CACHE_DIR", "hf_cache"))

# # Optional local LLM path or HF hub id (used by chat/rag when you enable local LLM)
# LLAMA_MODEL_PATH: str = os.environ.get("LLM_MODEL_PATH", "")

# # ---- embeddings / models -----------------------------------------------------

# # Sentence-Transformer for embeddings (small, fast, good quality)
# SENTENCE_TFORMER: str = os.environ.get(
#     "SENTENCE_TFORMER",
#     "sentence-transformers/all-MiniLM-L6-v2",
# )

# # ---- device selection --------------------------------------------------------

# # "auto" -> use CUDA if available; else CPU. You can force "cpu" or "cuda".
# DEVICE: str = os.environ.get("DEVICE", "auto").lower()

# # ---- Neo4j (Community Edition compatible) -----------------------------------

# # CE uses the default "neo4j" database; we keep NEO4J_DB for compatibility but don't require it.
# NEO4J_URI: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
# NEO4J_USER: str = os.environ.get("NEO4J_USER", "neo4j")
# NEO4J_PASS: str = os.environ.get("NEO4J_PASS", "neo4j")
# NEO4J_DB: str   = os.environ.get("NEO4J_DB", "neo4j")  # ignored by CE scripts but available

# # ---- ingest + retrieval knobs ------------------------------------------------

# # For text chunking into Chroma
# CHUNK_SIZE: int    = int(os.environ.get("CHUNK_SIZE", "800"))
# CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "120"))

# # Batch sizes
# BATCH_SIZE: int    = int(os.environ.get("BATCH_SIZE", "2000"))     # general batch
# CHROMA_BATCH: int  = int(os.environ.get("CHROMA_BATCH", "3000"))   # Chroma upsert threshold

# # Parallelism for ingest
# PARALLEL_JOBS: int = int(os.environ.get("PARALLEL_JOBS", "4"))

# # ---- runtime helpers ---------------------------------------------------------

# def device_is_cuda() -> bool:
#     if DEVICE == "cuda":
#         return True
#     if DEVICE == "cpu":
#         return False
#     # "auto"
#     try:
#         import torch  # noqa: WPS433
#         return torch.cuda.is_available()
#     except Exception:
#         return False

# def effective_device() -> str:
#     return "cuda" if device_is_cuda() else "cpu"

# def print_config_summary() -> None:
#     print("— CONFIG —")
#     print(f"PROJECT_ROOT     : {PROJECT_ROOT}")
#     print(f"SQLITE_DB        : {SQLITE_DB}")
#     print(f"CHROMA_DIR       : {CHROMA_DIR}")
#     print(f"HF_CACHE_DIR     : {HF_CACHE_DIR}")
#     print(f"SENTENCE_TFORMER : {SENTENCE_TFORMER}")
#     print(f"LLAMA_MODEL_PATH : {LLAMA_MODEL_PATH or '(disabled)'}")
#     print(f"DEVICE           : {DEVICE} → using {effective_device()}")
#     print(f"NEO4J_URI        : {NEO4J_URI}")
#     print(f"NEO4J_USER       : {NEO4J_USER}")
#     print(f"NEO4J_DB         : {NEO4J_DB} (ignored on CE)")
#     print(f"CHUNK_SIZE       : {CHUNK_SIZE}")
#     print(f"CHUNK_OVERLAP    : {CHUNK_OVERLAP}")
#     print(f"BATCH_SIZE       : {BATCH_SIZE}")
#     print(f"CHROMA_BATCH     : {CHROMA_BATCH}")
#     print(f"PARALLEL_JOBS    : {PARALLEL_JOBS}")
#     print("———————")


# config_full.py
import os

# --- SQLite (original full DB) ---
SQLITE_DB = os.getenv("SQLITE_DB", r"D:\OSPO\KG-RAG1\researchers_fixed.db")

# --- Neo4j ---
NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "OSPOlol@1234")
NEO4J_DB   = os.getenv("NEO4J_DB",   "syr-rag")   # CE will still use 'neo4j' internally; kept for compatibility

# --- Chroma ---
CHROMA_DIR = os.getenv("CHROMA_DIR", r"D:\OSPO\KG-RAG1\chroma_store_full")

# --- LLaMA (Transformers format dir; not GGUF) ---
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", r"D:\OSPO\KG-RAG1\Llama-3.2-1B-Instruct")

# --- Ingest controls ---
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "2000"))
CHROMA_BATCH  = int(os.getenv("CHROMA_BATCH", "3000"))
PARALLEL_JOBS = int(os.getenv("PARALLEL_JOBS", "4"))

# --- Embeddings ---
SENTENCE_TFORMER = os.getenv("SENTENCE_TFORMER", "sentence-transformers/all-MiniLM-L6-v2")
