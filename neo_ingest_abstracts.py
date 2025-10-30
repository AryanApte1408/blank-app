# # """
# # neo_ingest_abstracts.py
# # Builds a simplified Neo4j graph from abstracts_only.db:
# #   (Paper {doi, title, year})
# #   (Source {name})
# #   (Year {value})
# # Relationships:
# #   (Source)-[:PUBLISHED]->(Paper)
# #   (Year)-[:CONTAINS]->(Paper)
# # """

# # import os
# # import sqlite3
# # from neo4j import GraphDatabase
# # import config_full as config

# # # Use your centralized config path
# # DB_PATH = config.SQLITE_DB

# # driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # def read_rows():
# #     if not os.path.exists(DB_PATH):
# #         raise FileNotFoundError(f"❌ SQLite database not found: {DB_PATH}")
# #     conn = sqlite3.connect(DB_PATH)
# #     cur = conn.cursor()
# #     cur.execute("""
# #         SELECT doi, title, abstract, source, year
# #         FROM abstracts_only
# #         WHERE abstract IS NOT NULL AND abstract != ''
# #     """)
# #     rows = cur.fetchall()
# #     conn.close()
# #     return rows

# # def ensure_schema():
# #     with driver.session(database=config.NEO4J_DB) as s:
# #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE")
# #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE")
# #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE")

# # UPSERT = """
# # UNWIND $rows AS row
# # MERGE (p:Paper {doi: row.doi})
# #   SET p.title = row.title,
# #       p.year  = row.year

# # WITH p, row
# # WHERE row.source IS NOT NULL AND row.source <> ""
# # MERGE (s:Source {name: row.source})
# # MERGE (s)-[:PUBLISHED]->(p)

# # WITH p, row
# # WHERE row.year IS NOT NULL AND row.year <> ""
# # MERGE (y:Year {value: row.year})
# # MERGE (y)-[:CONTAINS]->(p)
# # """

# # def ingest(rows, batch_size=200):
# #     batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]
# #     with driver.session(database=config.NEO4J_DB) as s:
# #         for i, batch in enumerate(batches, 1):
# #             s.run(UPSERT, rows=[
# #                 {"doi": r[0], "title": r[1], "abstract": r[2], "source": r[3], "year": r[4]} for r in batch
# #             ])
# #             print(f"✅ Batch {i}/{len(batches)} committed ({len(batch)} records).")

# # def main():
# #     print("— Neo4j Abstracts Ingest —")
# #     print(f"SQLite: {DB_PATH}")
# #     print(f"Neo4j : {config.NEO4J_URI} / db={config.NEO4J_DB}")
# #     print("───────────────────────────────")

# #     rows = read_rows()
# #     print(f"Rows prepared: {len(rows)}")
# #     if not rows:
# #         print("⚠️ No rows found in abstracts_only.db")
# #         return

# #     ensure_schema()
# #     ingest(rows)
# #     print("✅ Done ingesting abstracts into Neo4j.")

# # if __name__ == "__main__":
# #     main()

# """
# neo_ingest_abstracts.py
# Builds an enriched Neo4j graph from abstracts_only.db:
#   (Paper {doi, title, abstract, year, info, publication_date})
#   (Source {name})
#   (Year {value})
#   (Researcher {name})
# Relationships:
#   (Source)-[:PUBLISHED]->(Paper)
#   (Year)-[:CONTAINS]->(Paper)
#   (Researcher)-[:AUTHORED]->(Paper)
# """

# import os
# import sqlite3
# from neo4j import GraphDatabase
# import config_full as config

# # ─────────────────────────────────────────────────────────────────────────────
# # CONFIGURATION
# # ─────────────────────────────────────────────────────────────────────────────
# DB_PATH = config.SQLITE_DB
# driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # ─────────────────────────────────────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────────────────────────────────────
# def read_rows():
#     """Read enriched abstracts from SQLite."""
#     if not os.path.exists(DB_PATH):
#         raise FileNotFoundError(f"❌ SQLite database not found: {DB_PATH}")

#     conn = sqlite3.connect(DB_PATH)
#     cur = conn.cursor()

#     # ensure upgraded schema
#     cur.execute("PRAGMA table_info(abstracts_only);")
#     cols = {r[1] for r in cur.fetchall()}
#     expected = {"doi", "title", "abstract", "source", "year",
#                 "researcher_name", "work_title", "authors", "info", "publication_date"}
#     missing = expected - cols
#     if missing:
#         raise ValueError(f"❌ Missing columns in abstracts_only.db: {missing}")

#     cur.execute("""
#         SELECT doi, title, abstract, source, year,
#                researcher_name, work_title, authors, info, publication_date
#         FROM abstracts_only
#         WHERE abstract IS NOT NULL AND abstract != ''
#     """)
#     rows = cur.fetchall()
#     conn.close()
#     return rows


# def ensure_schema():
#     """Create unique constraints for core node labels."""
#     with driver.session(database=config.NEO4J_DB) as s:
#         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE")
#         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE")
#         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE")
#         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")
#     print("✅ Schema constraints verified.")


# # ─────────────────────────────────────────────────────────────────────────────
# # CYPHER UPSERT STATEMENT
# # ─────────────────────────────────────────────────────────────────────────────
# UPSERT = """
# UNWIND $rows AS row
# MERGE (p:Paper {doi: row.doi})
#   SET p.title = row.title,
#       p.abstract = row.abstract,
#       p.year = row.year,
#       p.info = row.info,
#       p.publication_date = row.publication_date,
#       p.work_title = row.work_title

# WITH p, row
# WHERE row.source IS NOT NULL AND row.source <> ""
# MERGE (s:Source {name: row.source})
# MERGE (s)-[:PUBLISHED]->(p)

# WITH p, row
# WHERE row.year IS NOT NULL AND row.year <> ""
# MERGE (y:Year {value: row.year})
# MERGE (y)-[:CONTAINS]->(p)

# WITH p, row
# WHERE row.researcher_name IS NOT NULL AND row.researcher_name <> ""
# MERGE (r:Researcher {name: row.researcher_name})
# MERGE (r)-[:AUTHORED]->(p)
# """

# # ─────────────────────────────────────────────────────────────────────────────
# # INGESTION LOGIC
# # ─────────────────────────────────────────────────────────────────────────────
# def ingest(rows, batch_size=200):
#     """Batch upsert papers and relationships."""
#     batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
#     with driver.session(database=config.NEO4J_DB) as s:
#         for i, batch in enumerate(batches, 1):
#             s.run(UPSERT, rows=[
#                 {
#                     "doi": r[0],
#                     "title": r[1],
#                     "abstract": r[2],
#                     "source": r[3],
#                     "year": r[4],
#                     "researcher_name": r[5],
#                     "work_title": r[6],
#                     "authors": r[7],
#                     "info": r[8],
#                     "publication_date": r[9],
#                 }
#                 for r in batch
#             ])
#             print(f"✅ Batch {i}/{len(batches)} committed ({len(batch)} records).")


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────────────────
# def main():
#     print("— Neo4j Abstracts Ingest —")
#     print(f"SQLite: {DB_PATH}")
#     print(f"Neo4j : {config.NEO4J_URI} / db={config.NEO4J_DB}")
#     print("───────────────────────────────")

#     rows = read_rows()
#     print(f"📘 Rows prepared: {len(rows)}")
#     if not rows:
#         print("⚠️ No valid abstracts found in abstracts_only.db.")
#         return

#     ensure_schema()
#     ingest(rows)
#     print("✅ Done ingesting enriched abstracts into Neo4j.")


# if __name__ == "__main__":
#     main()


"""
neo_ingest_abstracts.py
Builds an enriched Neo4j graph from abstracts_only.db:
  (Paper {doi, title, abstract, year, info, publication_date})
  (Source {name})
  (Year {value})
  (Researcher {name})
Relationships:
  (Source)-[:PUBLISHED]->(Paper)
  (Year)-[:CONTAINS]->(Paper)
  (Researcher)-[:AUTHORED]->(Paper)
"""

import os
import sqlite3
from neo4j import GraphDatabase
import config_full as config

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = config.SQLITE_DB
driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def read_rows():
    """Read enriched abstracts from SQLite."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"❌ SQLite database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ensure upgraded schema
    cur.execute("PRAGMA table_info(abstracts_only);")
    cols = {r[1] for r in cur.fetchall()}
    expected = {"doi", "title", "abstract", "source", "year",
                "researcher_name", "work_title", "authors", "info", "publication_date"}
    missing = expected - cols
    if missing:
        raise ValueError(f"❌ Missing columns in abstracts_only.db: {missing}")

    cur.execute("""
        SELECT doi, title, abstract, source, year,
               researcher_name, work_title, authors, info, publication_date
        FROM abstracts_only
        WHERE abstract IS NOT NULL AND abstract != ''
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def ensure_schema():
    """Create unique constraints for core node labels."""
    with driver.session(database=config.NEO4J_DB) as s:
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")
    print("✅ Schema constraints verified.")


# ─────────────────────────────────────────────────────────────────────────────
# CYPHER UPSERT STATEMENT
# ─────────────────────────────────────────────────────────────────────────────
UPSERT = """
UNWIND $rows AS row
MERGE (p:Paper {doi: row.doi})
  SET p.title = row.title,
      p.abstract = row.abstract,
      p.year = row.year,
      p.info = row.info,
      p.publication_date = row.publication_date,
      p.work_title = row.work_title

WITH p, row
WHERE row.source IS NOT NULL AND row.source <> ""
MERGE (s:Source {name: row.source})
MERGE (s)-[:PUBLISHED]->(p)

WITH p, row
WHERE row.year IS NOT NULL AND row.year <> ""
MERGE (y:Year {value: row.year})
MERGE (y)-[:CONTAINS]->(p)

WITH p, row
WHERE row.researcher_name IS NOT NULL AND row.researcher_name <> ""
MERGE (r:Researcher {name: row.researcher_name})
MERGE (r)-[:AUTHORED]->(p)
"""

# ─────────────────────────────────────────────────────────────────────────────
# INGESTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def ingest(rows, batch_size=200):
    """Batch upsert papers and relationships."""
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    with driver.session(database=config.NEO4J_DB) as s:
        for i, batch in enumerate(batches, 1):
            s.run(UPSERT, rows=[
                {
                    "doi": r[0],
                    "title": r[1],
                    "abstract": r[2],
                    "source": r[3],
                    "year": r[4],
                    "researcher_name": r[5],
                    "work_title": r[6],
                    "authors": r[7],
                    "info": r[8],
                    "publication_date": r[9],
                }
                for r in batch
            ])
            print(f"✅ Batch {i}/{len(batches)} committed ({len(batch)} records).")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("— Neo4j Abstracts Ingest —")
    print(f"SQLite: {DB_PATH}")
    print(f"Neo4j : {config.NEO4J_URI} / db={config.NEO4J_DB}")
    print("───────────────────────────────")

    rows = read_rows()
    print(f"📘 Rows prepared: {len(rows)}")
    if not rows:
        print("⚠️ No valid abstracts found in abstracts_only.db.")
        return

    ensure_schema()
    ingest(rows)
    print("✅ Done ingesting enriched abstracts into Neo4j.")


if __name__ == "__main__":
    main()
