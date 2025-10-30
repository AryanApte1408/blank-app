# # # import os

# # # #SQLITE_DB = os.getenv("SQLITE_DB", r"D:\OSPO\KG-RAG1\researchers_fixed.db")
# # # SQLITE_DB = os.getenv("SQLITE_DB", r"D:\OSPO\KG-RAG1\abstracts_only_fix.db")

# # # NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
# # # NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# # # NEO4J_PASS = os.getenv("NEO4J_PASS", "OSPOlol@1234")
# # # NEO4J_DB   = os.getenv("NEO4J_DB",   "syr-rag-abstracts")

# # # # CHROMA_DIR = os.getenv("CHROMA_DIR", r"D:\OSPO\KG-RAG1\chroma_store_full")
# # # CHROMA_DIR = os.getenv("CHROMA_DIR", r"D:\OSPO\KG-RAG1\chroma_store_abstracts")

# # # LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", r"D:\OSPO\KG-RAG1\Llama-3.2-1B-Instruct")

# # # CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
# # # CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
# # # BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "2000"))
# # # CHROMA_BATCH  = int(os.getenv("CHROMA_BATCH", "3000"))
# # # PARALLEL_JOBS = int(os.getenv("PARALLEL_JOBS", "4"))

# # # SENTENCE_TFORMER = os.getenv("SENTENCE_TFORMER", "sentence-transformers/all-MiniLM-L6-v2")

# # """
# # config_full.py - Lightweight config that delegates to DatabaseManager
# # """
# # import os
# # from database_manager import get_active_db_config, get_db_manager

# # # Model configuration (independent of databases)
# # LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", r"D:\OSPO\KG-RAG1\Llama-3.2-1B-Instruct")

# # # Processing parameters (independent of databases)
# # CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
# # CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
# # BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "2000"))
# # CHROMA_BATCH  = int(os.getenv("CHROMA_BATCH", "3000"))
# # PARALLEL_JOBS = int(os.getenv("PARALLEL_JOBS", "4"))

# # SENTENCE_TFORMER = os.getenv("SENTENCE_TFORMER", "sentence-transformers/all-MiniLM-L6-v2")


# # # Dynamic properties that read from active database config
# # @property
# # def SQLITE_DB():
# #     return get_active_db_config().sqlite_path

# # @property
# # def CHROMA_DIR():
# #     return get_active_db_config().chroma_dir

# # @property
# # def NEO4J_URI():
# #     return get_active_db_config().neo4j_uri

# # @property
# # def NEO4J_USER():
# #     return get_active_db_config().neo4j_user

# # @property
# # def NEO4J_PASS():
# #     return get_active_db_config().neo4j_password

# # @property
# # def NEO4J_DB():
# #     return get_active_db_config().neo4j_database


# # # Compatibility aliases
# # DB_PATH = SQLITE_DB
# # SQLITE_DB_FULL = SQLITE_DB
# # SQLITE_DB_ABSTRACTS = SQLITE_DB
# # CHROMA_DIR_FULL = CHROMA_DIR
# # CHROMA_DIR_ABSTRACTS = CHROMA_DIR


# # def get_active_mode():
# #     """Get the current database mode."""
# #     return get_active_db_config().mode


# # def get_active_collection():
# #     """Get the current ChromaDB collection name."""
# #     return get_active_db_config().chroma_collection

# """
# config_full.py - Lightweight config (NO SQLite dependencies)
# """
# import os
# from database_manager import get_active_db_config, get_db_manager

# # Model configuration (independent of databases)
# LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", r"D:\OSPO\KG-RAG1\Llama-3.2-1B-Instruct")

# # Processing parameters (independent of databases)
# CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
# CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
# BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "2000"))
# CHROMA_BATCH  = int(os.getenv("CHROMA_BATCH", "3000"))
# PARALLEL_JOBS = int(os.getenv("PARALLEL_JOBS", "4"))

# SENTENCE_TFORMER = os.getenv("SENTENCE_TFORMER", "sentence-transformers/all-MiniLM-L6-v2")


# # Dynamic properties that read from active database config (NO SQLite!)
# def get_chroma_dir():
#     return get_active_db_config().chroma_dir

# def get_neo4j_uri():
#     return get_active_db_config().neo4j_uri

# def get_neo4j_user():
#     return get_active_db_config().neo4j_user

# def get_neo4j_pass():
#     return get_active_db_config().neo4j_password

# def get_neo4j_db():
#     return get_active_db_config().neo4j_database

# def get_active_mode():
#     """Get the current database mode."""
#     return get_active_db_config().mode

# def get_active_collection():
#     """Get the current ChromaDB collection name."""
#     return get_active_db_config().chroma_collection


# # Legacy aliases (kept for backwards compatibility, but point to ChromaDB only)
# CHROMA_DIR_FULL = os.getenv("CHROMA_DIR_FULL", r"D:\OSPO\KG-RAG1\chroma_store_full")
# CHROMA_DIR_ABSTRACTS = os.getenv("CHROMA_DIR_ABSTRACTS", r"D:\OSPO\KG-RAG1\chroma_store_abstracts")
"""
config_full.py - Lightweight config (NO SQLite, NO hardcoded paths in RAG)
"""
import os
from database_manager import get_active_db_config

# Model configuration
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", r"D:\OSPO\KG-RAG1\Llama-3.2-1B-Instruct")

# Processing parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2000"))
CHROMA_BATCH = int(os.getenv("CHROMA_BATCH", "3000"))
PARALLEL_JOBS = int(os.getenv("PARALLEL_JOBS", "4"))

SENTENCE_TFORMER = os.getenv("SENTENCE_TFORMER", "sentence-transformers/all-MiniLM-L6-v2")