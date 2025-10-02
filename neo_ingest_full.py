# # # # # # # # # # # import re, sqlite3
# # # # # # # # # # # from typing import List, Dict
# # # # # # # # # # # from neo4j import GraphDatabase
# # # # # # # # # # # from tqdm import tqdm
# # # # # # # # # # # import config_full as config

# # # # # # # # # # # driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # # # # # # # # # # DATE_RE = re.compile(r"(19|20)\d{2}")
# # # # # # # # # # # PUBDATE_RE = re.compile(r"Publication Date:\s*([0-9]{4})(?:-[0-9]{2}-[0-9]{2})?", re.I)
# # # # # # # # # # # DOI_RE  = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
# # # # # # # # # # # URL_RE  = re.compile(r"https?://\S+")

# # # # # # # # # # # def extract_year(info: str):
# # # # # # # # # # #     if not info: return None
# # # # # # # # # # #     m = PUBDATE_RE.search(info)
# # # # # # # # # # #     if m: return int(m.group(1))
# # # # # # # # # # #     m2 = DATE_RE.search(info)
# # # # # # # # # # #     return int(m2.group(0)) if m2 else None

# # # # # # # # # # # def extract_doi_or_url(info: str):
# # # # # # # # # # #     if not info: return None
# # # # # # # # # # #     m = DOI_RE.search(info)
# # # # # # # # # # #     if m: return f"https://doi.org/{m.group(1)}"
# # # # # # # # # # #     m2 = URL_RE.search(info)
# # # # # # # # # # #     return m2.group(0) if m2 else None

# # # # # # # # # # # def split_authors(authors: str, researcher: str) -> list[str]:
# # # # # # # # # # #     if not authors and researcher:
# # # # # # # # # # #         return [researcher.strip()]
# # # # # # # # # # #     if not authors:
# # # # # # # # # # #         return []
# # # # # # # # # # #     cleaned = authors.replace(" and ", ";").replace(",", ";").replace("|", ";")
# # # # # # # # # # #     return [a.strip() for a in cleaned.split(";") if a.strip()]

# # # # # # # # # # # def read_rows_from_sqlite() -> List[Dict]:
# # # # # # # # # # #     conn = sqlite3.connect(config.SQLITE_DB)
# # # # # # # # # # #     cur  = conn.cursor()
# # # # # # # # # # #     cur.execute("SELECT id, researcher_name, work_title, authors, info FROM research_info")
# # # # # # # # # # #     raw = cur.fetchall()
# # # # # # # # # # #     conn.close()

# # # # # # # # # # #     rows = []
# # # # # # # # # # #     for pid, researcher, title, authors, info in raw:
# # # # # # # # # # #         if not title:
# # # # # # # # # # #             continue
# # # # # # # # # # #         author_list = split_authors(authors, researcher)
# # # # # # # # # # #         year = extract_year(info or "")
# # # # # # # # # # #         doi  = extract_doi_or_url(info or "")
# # # # # # # # # # #         rows.append({
# # # # # # # # # # #             "paper_id": str(pid),
# # # # # # # # # # #             "title": title or "",
# # # # # # # # # # #             "info": info or "",
# # # # # # # # # # #             "researcher_name": researcher or "",
# # # # # # # # # # #             "authors": author_list,
# # # # # # # # # # #             "year": year,
# # # # # # # # # # #             "doi_link": doi,
# # # # # # # # # # #         })
# # # # # # # # # # #     return rows

# # # # # # # # # # # UPSERT = """
# # # # # # # # # # # UNWIND $rows AS row
# # # # # # # # # # # MERGE (p:Paper {paper_id: row.paper_id})
# # # # # # # # # # #   SET p.title=row.title, p.info=row.info, p.year=row.year, p.doi_link=row.doi_link

# # # # # # # # # # # WITH p, row
# # # # # # # # # # # WHERE row.researcher_name IS NOT NULL AND row.researcher_name <> ""
# # # # # # # # # # # MERGE (r:Researcher {name: row.researcher_name})
# # # # # # # # # # # MERGE (r)-[:WROTE]->(p)
# # # # # # # # # # # MERGE (p)-[:HAS_RESEARCHER]->(r)

# # # # # # # # # # # WITH p, row
# # # # # # # # # # # UNWIND row.authors AS authorName
# # # # # # # # # # # MERGE (a:Author {name: authorName})
# # # # # # # # # # # MERGE (a)-[:AUTHORED]->(p)
# # # # # # # # # # # MERGE (p)-[:HAS_AUTHOR]->(a)

# # # # # # # # # # # WITH p, collect(DISTINCT a) AS authors
# # # # # # # # # # # UNWIND authors AS a1
# # # # # # # # # # # UNWIND authors AS a2
# # # # # # # # # # # WITH a1, a2 WHERE id(a1) < id(a2)
# # # # # # # # # # # MERGE (a1)-[:COAUTHORED_WITH]-(a2)
# # # # # # # # # # # """

# # # # # # # # # # # def ingest(rows):
# # # # # # # # # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # # # # # # # # #         for i in tqdm(range(0, len(rows), 2000), desc="Neo4j ingest"):
# # # # # # # # # # #             s.run(UPSERT, rows=rows[i:i+2000])

# # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # #     rows = read_rows_from_sqlite()
# # # # # # # # # # #     print(f"📦 Rows to ingest: {len(rows)}")
# # # # # # # # # # #     ingest(rows)
# # # # # # # # # # #     print("✅ Neo4j full ingestion complete (all features, all authors, all links)")
# # # # # # # # # # # neo4_ingest_full.py
# # # # # # # # # # import re
# # # # # # # # # # import sqlite3
# # # # # # # # # # from typing import List, Dict, Any
# # # # # # # # # # from neo4j import GraphDatabase
# # # # # # # # # # from tqdm import tqdm
# # # # # # # # # # import config_full as config

# # # # # # # # # # # --- Neo4j Driver ---
# # # # # # # # # # driver = GraphDatabase.driver(
# # # # # # # # # #     config.NEO4J_URI,
# # # # # # # # # #     auth=(config.NEO4J_USER, config.NEO4J_PASS)
# # # # # # # # # # )

# # # # # # # # # # # --- Regex helpers ---
# # # # # # # # # # DATE_RE    = re.compile(r"\b(19|20)\d{2}\b")
# # # # # # # # # # PUBDATE_RE = re.compile(r"Publication Date:\s*([0-9]{4})", re.I)
# # # # # # # # # # DOI_RE     = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
# # # # # # # # # # URL_RE     = re.compile(r"https?://\S+")
# # # # # # # # # # SEP_RX     = re.compile(r"\s*(?:;|,|\band\b|\&|\|)\s*", re.I)

# # # # # # # # # # def extract_year(info: str) -> int | None:
# # # # # # # # # #     """Extract publication year from info string."""
# # # # # # # # # #     if not info:
# # # # # # # # # #         return None
# # # # # # # # # #     if (m := PUBDATE_RE.search(info)):
# # # # # # # # # #         return int(m.group(1))
# # # # # # # # # #     if (m2 := DATE_RE.search(info)):
# # # # # # # # # #         return int(m2.group(0))
# # # # # # # # # #     return None

# # # # # # # # # # def extract_link(info: str) -> str | None:
# # # # # # # # # #     """Extract DOI or URL from info string."""
# # # # # # # # # #     if not info:
# # # # # # # # # #         return None
# # # # # # # # # #     if (m := DOI_RE.search(info)):
# # # # # # # # # #         return f"https://doi.org/{m.group(0)}"
# # # # # # # # # #     if (u := URL_RE.search(info)):
# # # # # # # # # #         return u.group(0)
# # # # # # # # # #     return None

# # # # # # # # # # def parse_authors(authors: str, researcher_name: str) -> tuple[str, list[str]]:
# # # # # # # # # #     """
# # # # # # # # # #     Returns (primary_author, co_authors[]).
# # # # # # # # # #     - primary_author: researcher_name if present, else first in authors list
# # # # # # # # # #     - co_authors: rest (deduped, excluding primary)
# # # # # # # # # #     """
# # # # # # # # # #     pool: list[str] = []
# # # # # # # # # #     if authors:
# # # # # # # # # #         pool = [a.strip() for a in SEP_RX.split(authors) if a.strip()]
# # # # # # # # # #     primary = (researcher_name or "").strip() or (pool[0] if pool else "")
# # # # # # # # # #     co = [a for a in pool if a and a.lower() != primary.lower()]
# # # # # # # # # #     seen, co_unique = set(), []
# # # # # # # # # #     for a in co:
# # # # # # # # # #         k = a.lower()
# # # # # # # # # #         if k not in seen:
# # # # # # # # # #             seen.add(k)
# # # # # # # # # #             co_unique.append(a)
# # # # # # # # # #     return primary, co_unique

# # # # # # # # # # def read_rows() -> List[Dict[str, Any]]:
# # # # # # # # # #     """Read rows from SQLite and normalize metadata."""
# # # # # # # # # #     conn = sqlite3.connect(config.SQLITE_DB)
# # # # # # # # # #     cur  = conn.cursor()
# # # # # # # # # #     cur.execute("SELECT id, researcher_name, work_title, authors, info FROM research_info")
# # # # # # # # # #     raw = cur.fetchall()
# # # # # # # # # #     conn.close()

# # # # # # # # # #     rows = []
# # # # # # # # # #     for pid, rname, title, authors, info in raw:
# # # # # # # # # #         if not title:
# # # # # # # # # #             continue
# # # # # # # # # #         primary, co = parse_authors(authors or "", rname or "")
# # # # # # # # # #         rows.append({
# # # # # # # # # #             "paper_id": str(pid),
# # # # # # # # # #             "title": title.strip(),
# # # # # # # # # #             "info": (info or "").strip(),
# # # # # # # # # #             "primary_author": primary,
# # # # # # # # # #             "co_authors": co,
# # # # # # # # # #             "year": extract_year(info or ""),
# # # # # # # # # #             "doi_link": extract_link(info or "")
# # # # # # # # # #         })
# # # # # # # # # #     return rows

# # # # # # # # # # def ensure_schema():
# # # # # # # # # #     """Ensure Neo4j constraints & indexes exist."""
# # # # # # # # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # # # # # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
# # # # # # # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
# # # # # # # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")
# # # # # # # # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.year)")

# # # # # # # # # # # --- Cypher UPSERT ---
# # # # # # # # # # UPSERT = """
# # # # # # # # # # UNWIND $rows AS row
# # # # # # # # # # MERGE (p:Paper {paper_id: row.paper_id})
# # # # # # # # # #   SET p.title = row.title,
# # # # # # # # # #       p.info = row.info,
# # # # # # # # # #       p.year = row.year,
# # # # # # # # # #       p.doi_link = row.doi_link

# # # # # # # # # # // Primary author
# # # # # # # # # # WITH p, row
# # # # # # # # # # WHERE row.primary_author IS NOT NULL AND row.primary_author <> ""
# # # # # # # # # # MERGE (r:Researcher {name: row.primary_author})
# # # # # # # # # # MERGE (a_primary:Author {name: row.primary_author})
# # # # # # # # # # MERGE (r)-[:WROTE]->(p)
# # # # # # # # # # MERGE (a_primary)-[:AUTHORED]->(p)
# # # # # # # # # # MERGE (p)-[:HAS_RESEARCHER]->(r)
# # # # # # # # # # MERGE (p)-[:HAS_AUTHOR]->(a_primary)

# # # # # # # # # # // Co-authors
# # # # # # # # # # WITH p, row, a_primary
# # # # # # # # # # UNWIND row.co_authors AS coName
# # # # # # # # # # MERGE (a:Author {name: coName})
# # # # # # # # # # MERGE (a)-[:AUTHORED]->(p)
# # # # # # # # # # MERGE (p)-[:HAS_AUTHOR]->(a)

# # # # # # # # # # // Co-author network (avoid id() deprecation)
# # # # # # # # # # WITH p
# # # # # # # # # # MATCH (p)-[:HAS_AUTHOR]->(a1:Author)
# # # # # # # # # # MATCH (p)-[:HAS_AUTHOR]->(a2:Author)
# # # # # # # # # # WHERE a1.name < a2.name
# # # # # # # # # # MERGE (a1)-[:COAUTHORED_WITH]-(a2)
# # # # # # # # # # """

# # # # # # # # # # def ingest(rows: List[Dict[str, Any]]):
# # # # # # # # # #     """Batch ingest rows into Neo4j."""
# # # # # # # # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # # # # # # # #         for i in tqdm(range(0, len(rows), config.BATCH_SIZE), desc="Neo4j ingest"):
# # # # # # # # # #             batch = rows[i:i+config.BATCH_SIZE]
# # # # # # # # # #             s.run(UPSERT, rows=batch)

# # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # #     ensure_schema()
# # # # # # # # # #     rows = read_rows()
# # # # # # # # # #     print(f"📦 Rows to ingest: {len(rows)}")
# # # # # # # # # #     ingest(rows)
# # # # # # # # # #     print("✅ Neo4j full ingestion complete.")

# # # # # # # # # import re
# # # # # # # # # import sqlite3
# # # # # # # # # from typing import List, Dict, Any
# # # # # # # # # from neo4j import GraphDatabase
# # # # # # # # # from tqdm import tqdm
# # # # # # # # # import config_full as config

# # # # # # # # # driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # # # # # # # # # --- Regex helpers ---
# # # # # # # # # DATE_RE    = re.compile(r"\b(19|20)\d{2}\b")
# # # # # # # # # PUBDATE_RE = re.compile(r"Publication Date:\s*([0-9]{4})(?:-[0-9]{2}-[0-9]{2})?", re.I)
# # # # # # # # # DOI_RE     = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
# # # # # # # # # URL_RE     = re.compile(r"https?://\S+")
# # # # # # # # # SEP_RX     = re.compile(r"\s*(?:;|,|\band\b|&|\|)\s*")

# # # # # # # # # # ----------------- Helpers -----------------

# # # # # # # # # def extract_year(info: str) -> str:
# # # # # # # # #     if not info:
# # # # # # # # #         return None
# # # # # # # # #     m = PUBDATE_RE.search(info)
# # # # # # # # #     if m:
# # # # # # # # #         return m.group(1)
# # # # # # # # #     m = DATE_RE.search(info)
# # # # # # # # #     if m:
# # # # # # # # #         return m.group(0)
# # # # # # # # #     return None

# # # # # # # # # def extract_link(info: str) -> str:
# # # # # # # # #     if not info:
# # # # # # # # #         return None
# # # # # # # # #     m = DOI_RE.search(info)
# # # # # # # # #     if m:
# # # # # # # # #         return f"https://doi.org/{m.group(0)}"
# # # # # # # # #     m = URL_RE.search(info)
# # # # # # # # #     if m:
# # # # # # # # #         return m.group(0)
# # # # # # # # #     return None

# # # # # # # # # def read_rows() -> List[Dict[str, Any]]:
# # # # # # # # #     """Read rows from SQLite research_info and normalize/dedup authors."""
# # # # # # # # #     conn = sqlite3.connect(config.SQLITE_DB)
# # # # # # # # #     cur = conn.cursor()
# # # # # # # # #     cur.execute("SELECT id, researcher_name, work_title, authors, info FROM research_info")
# # # # # # # # #     raw = cur.fetchall()
# # # # # # # # #     conn.close()

# # # # # # # # #     rows = []
# # # # # # # # #     for pid, rname, title, authors, info in raw:
# # # # # # # # #         if not title:
# # # # # # # # #             continue

# # # # # # # # #         # Split + dedup authors
# # # # # # # # #         pool = [a.strip() for a in SEP_RX.split(authors or "") if a.strip()]
# # # # # # # # #         seen, author_list = set(), []
# # # # # # # # #         for a in pool:
# # # # # # # # #             key = a.lower()
# # # # # # # # #             if key not in seen:
# # # # # # # # #                 seen.add(key)
# # # # # # # # #                 author_list.append(a)

# # # # # # # # #         rows.append({
# # # # # # # # #             "paper_id": str(pid),
# # # # # # # # #             "title": title or "",
# # # # # # # # #             "info": info or "",
# # # # # # # # #             "researcher_name": rname or "",
# # # # # # # # #             "authors": author_list,
# # # # # # # # #             "year": extract_year(info or ""),
# # # # # # # # #             "doi": extract_link(info or "")
# # # # # # # # #         })
# # # # # # # # #     return rows

# # # # # # # # # # ----------------- Ingestion -----------------

# # # # # # # # # def ensure_schema():
# # # # # # # # #     """Create uniqueness constraints to avoid duplicate nodes."""
# # # # # # # # #     with driver.session(database=config.NEO4J_DB) as session:
# # # # # # # # #         session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
# # # # # # # # #         session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
# # # # # # # # #         session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")

# # # # # # # # # def ingest(rows: List[Dict[str, Any]]):
# # # # # # # # #     """Ingest rows into Neo4j with MERGE to avoid duplicates."""
# # # # # # # # #     query = """
# # # # # # # # #     MERGE (p:Paper {paper_id: $paper_id})
# # # # # # # # #       ON CREATE SET p.title = $title, p.info = $info, p.year = $year, p.doi = $doi
# # # # # # # # #       ON MATCH SET p.title = COALESCE(p.title, $title),
# # # # # # # # #                     p.info  = COALESCE(p.info, $info),
# # # # # # # # #                     p.year  = COALESCE(p.year, $year),
# # # # # # # # #                     p.doi   = COALESCE(p.doi, $doi)

# # # # # # # # #     WITH p, $authors AS authors
# # # # # # # # #     UNWIND authors AS author_name
# # # # # # # # #       MERGE (a:Author {name: author_name})
# # # # # # # # #       MERGE (p)-[:HAS_AUTHOR]->(a)

# # # # # # # # #     WITH p, $researcher_name AS rname
# # # # # # # # #     WHERE rname IS NOT NULL AND rname <> ""
# # # # # # # # #       MERGE (r:Researcher {name: rname})
# # # # # # # # #       MERGE (p)-[:HAS_RESEARCHER]->(r)
# # # # # # # # #     """
# # # # # # # # #     with driver.session(database=config.NEO4J_DB) as session:
# # # # # # # # #         for row in tqdm(rows, desc="Ingesting"):
# # # # # # # # #             session.run(query, **row)

# # # # # # # # # def build_coauthor_edges():
# # # # # # # # #     """Create COAUTHORED_WITH links between authors of the same paper."""
# # # # # # # # #     query = """
# # # # # # # # #     MATCH (p:Paper)<-[:HAS_AUTHOR]-(a1:Author),
# # # # # # # # #           (p)<-[:HAS_AUTHOR]-(a2:Author)
# # # # # # # # #     WHERE id(a1) < id(a2)
# # # # # # # # #     MERGE (a1)-[:COAUTHORED_WITH]-(a2);
# # # # # # # # #     """
# # # # # # # # #     with driver.session(database=config.NEO4J_DB) as session:
# # # # # # # # #         session.run(query)
# # # # # # # # #     print("✅ Co-author relationships built.")

# # # # # # # # # # ----------------- Main -----------------

# # # # # # # # # if __name__ == "__main__":
# # # # # # # # #     ensure_schema()
# # # # # # # # #     rows = read_rows()
# # # # # # # # #     print(f"Read {len(rows)} rows from SQLite.")
# # # # # # # # #     ingest(rows)
# # # # # # # # #     build_coauthor_edges()
# # # # # # # # #     print("Done.")


# # # # # # # # # neo_ingest_full.py (safe batching)
# # # # # # # # import json, math, re, sqlite3
# # # # # # # # from typing import Any, Dict, List, Optional

# # # # # # # # from neo4j import GraphDatabase
# # # # # # # # from neo4j.exceptions import ClientError
# # # # # # # # from tqdm import tqdm

# # # # # # # # import config_full as config

# # # # # # # # # ────────────────────────────── Regex helpers
# # # # # # # # DATE_RE    = re.compile(r"\b(19|20)\d{2}\b")
# # # # # # # # PUBDATE_RE = re.compile(r"\b(?:Publication\s*Date|Date)\s*:\s*([0-9]{4})", re.I)
# # # # # # # # DOI_RE     = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
# # # # # # # # URL_RE     = re.compile(r"https?://\S+")
# # # # # # # # SEP_RX     = re.compile(r"\s*(?:;|,|\band\b|&|\|)\s*", re.I)

# # # # # # # # driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # # # # # # # # ────────────────────────────── Helpers
# # # # # # # # def safe_str(x: Any) -> str:
# # # # # # # #     if x is None:
# # # # # # # #         return ""
# # # # # # # #     if isinstance(x, float):
# # # # # # # #         try:
# # # # # # # #             if math.isnan(x): return ""
# # # # # # # #         except Exception:
# # # # # # # #             return ""
# # # # # # # #     return str(x).strip()

# # # # # # # # def extract_year(info: str) -> Optional[int]:
# # # # # # # #     s = safe_str(info)
# # # # # # # #     if not s: return None
# # # # # # # #     m = PUBDATE_RE.search(s)
# # # # # # # #     if m:
# # # # # # # #         try: return int(m.group(1))
# # # # # # # #         except: pass
# # # # # # # #     m2 = DATE_RE.search(s)
# # # # # # # #     return int(m2.group(0)) if m2 else None

# # # # # # # # def extract_doi_link(info: str, doi: str) -> Optional[str]:
# # # # # # # #     doi = safe_str(doi)
# # # # # # # #     if doi:
# # # # # # # #         return f"https://doi.org/{doi}" if doi.lower().startswith("10.") else doi
# # # # # # # #     s = safe_str(info)
# # # # # # # #     if not s: return None
# # # # # # # #     m = DOI_RE.search(s)
# # # # # # # #     if m: return f"https://doi.org/{m.group(0)}"
# # # # # # # #     m2 = URL_RE.search(s)
# # # # # # # #     return m2.group(0) if m2 else None

# # # # # # # # def parse_authors(cell: Any) -> List[str]:
# # # # # # # #     s = safe_str(cell)
# # # # # # # #     if not s: return []
# # # # # # # #     try:
# # # # # # # #         if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
# # # # # # # #             v = json.loads(s)
# # # # # # # #             if isinstance(v, list): return _dedupe([safe_str(x) for x in v if safe_str(x)])
# # # # # # # #             if isinstance(v, dict) and isinstance(v.get("authors"), list):
# # # # # # # #                 return _dedupe([safe_str(x) for x in v["authors"] if safe_str(x)])
# # # # # # # #     except Exception: pass
# # # # # # # #     return _dedupe([p for p in SEP_RX.split(s) if p])

# # # # # # # # def _dedupe(items: List[str]) -> List[str]:
# # # # # # # #     seen, out = set(), []
# # # # # # # #     for it in items:
# # # # # # # #         k = it.lower()
# # # # # # # #         if k and k not in seen:
# # # # # # # #             seen.add(k)
# # # # # # # #             out.append(it)
# # # # # # # #     return out

# # # # # # # # # ────────────────────────────── Read rows
# # # # # # # # def read_rows() -> List[Dict[str, Any]]:
# # # # # # # #     conn = sqlite3.connect(config.SQLITE_DB)
# # # # # # # #     cur = conn.cursor()
# # # # # # # #     cur.execute("PRAGMA table_info(research_info);")
# # # # # # # #     cols = {r[1] for r in cur.fetchall()}
# # # # # # # #     has_doi, has_pub = "doi" in cols, "publication_date" in cols

# # # # # # # #     sql = "SELECT id, researcher_name, work_title, authors, info"
# # # # # # # #     sql += ", doi" if has_doi else ", '' AS doi"
# # # # # # # #     sql += ", publication_date" if has_pub else ", '' AS publication_date"
# # # # # # # #     sql += " FROM research_info"
# # # # # # # #     cur.execute(sql)
# # # # # # # #     raw = cur.fetchall()
# # # # # # # #     conn.close()

# # # # # # # #     rows = []
# # # # # # # #     for pid, rname, title, authors, info, doi, pubdate in raw:
# # # # # # # #         title = safe_str(title)
# # # # # # # #         if not title: continue
# # # # # # # #         authors_list = parse_authors(authors)
# # # # # # # #         primary = safe_str(rname) or (authors_list[0] if authors_list else "")
# # # # # # # #         title_short = title[:2048]
# # # # # # # #         rows.append({
# # # # # # # #             "paper_id": str(pid),
# # # # # # # #             "title": title,
# # # # # # # #             "title_short": title_short,
# # # # # # # #             "info": safe_str(info),
# # # # # # # #             "primary_author": primary,
# # # # # # # #             "co_authors": [a for a in authors_list if a and a.lower() != primary.lower()],
# # # # # # # #             "year": extract_year(safe_str(pubdate) or safe_str(info)),
# # # # # # # #             "doi_link": extract_doi_link(info, doi),
# # # # # # # #         })
# # # # # # # #     return rows

# # # # # # # # # ────────────────────────────── Schema
# # # # # # # # def ensure_schema():
# # # # # # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # # # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
# # # # # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
# # # # # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")
# # # # # # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.year)")
# # # # # # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.doi_link)")
# # # # # # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.title_short)")

# # # # # # # # # ────────────────────────────── Cypher
# # # # # # # # UPSERT = """
# # # # # # # # UNWIND $rows AS row
# # # # # # # # MERGE (p:Paper {paper_id: row.paper_id})
# # # # # # # #   SET p.title       = row.title,
# # # # # # # #       p.title_short = row.title_short,
# # # # # # # #       p.info        = row.info,
# # # # # # # #       p.year        = row.year,
# # # # # # # #       p.doi_link    = row.doi_link

# # # # # # # # WITH p, row
# # # # # # # # WHERE row.primary_author IS NOT NULL AND row.primary_author <> ""
# # # # # # # # MERGE (r:Researcher {name: row.primary_author})
# # # # # # # # MERGE (a_primary:Author {name: row.primary_author})
# # # # # # # # MERGE (r)-[:WROTE]->(p)
# # # # # # # # MERGE (a_primary)-[:AUTHORED]->(p)
# # # # # # # # MERGE (p)-[:HAS_RESEARCHER]->(r)
# # # # # # # # MERGE (p)-[:HAS_AUTHOR]->(a_primary)

# # # # # # # # WITH p, row
# # # # # # # # UNWIND row.co_authors AS coName
# # # # # # # # WITH p, trim(coName) AS cname
# # # # # # # # WHERE cname <> ""
# # # # # # # # MERGE (a:Author {name: cname})
# # # # # # # # MERGE (a)-[:AUTHORED]->(p)
# # # # # # # # MERGE (p)-[:HAS_AUTHOR]->(a)
# # # # # # # # """

# # # # # # # # COAUTHORS = """
# # # # # # # # MATCH (p:Paper)<-[:HAS_AUTHOR]-(a1:Author),
# # # # # # # #       (p)<-[:HAS_AUTHOR]-(a2:Author)
# # # # # # # # WHERE id(a1) < id(a2)
# # # # # # # # MERGE (a1)-[:COAUTHORED_WITH]-(a2)
# # # # # # # # """

# # # # # # # # # ────────────────────────────── Ingest
# # # # # # # # def ingest(rows: List[Dict[str, Any]]):
# # # # # # # #     batch = min(getattr(config, "BATCH_SIZE", 200), 200)  # cap at 200
# # # # # # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # # # # # #         for i in tqdm(range(0, len(rows), batch), desc="Neo4j ingest"):
# # # # # # # #             chunk = rows[i:i+batch]
# # # # # # # #             s.execute_write(lambda tx: tx.run(UPSERT, rows=chunk))
# # # # # # # #             print(f"  ✓ Committed {i+len(chunk)}/{len(rows)}")

# # # # # # # # def build_coauthor_edges():
# # # # # # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # # # # # #         s.run(COAUTHORS)
# # # # # # # #     print("✅ Co-author edges built.")

# # # # # # # # # ────────────────────────────── Main
# # # # # # # # if __name__ == "__main__":
# # # # # # # #     try:
# # # # # # # #         ensure_schema()
# # # # # # # #     except ClientError as e:
# # # # # # # #         print("Schema setup failed:", e)
# # # # # # # #         raise

# # # # # # # #     rows = read_rows()
# # # # # # # #     print(f"📦 Rows prepared: {len(rows)}")
# # # # # # # #     ingest(rows)
# # # # # # # #     build_coauthor_edges()
# # # # # # # #     print("✅ Done.")


# # # # # # # # neo_ingest_full.py (with semantic embeddings)
# # # # # # # import json, math, re, sqlite3
# # # # # # # from typing import Any, Dict, List, Optional

# # # # # # # from neo4j import GraphDatabase
# # # # # # # from neo4j.exceptions import ClientError
# # # # # # # from tqdm import tqdm
# # # # # # # from sentence_transformers import SentenceTransformer

# # # # # # # import config_full as config

# # # # # # # # ────────────────────────────── Embedding model
# # # # # # # embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # # # # # # # ────────────────────────────── Regex helpers
# # # # # # # DATE_RE    = re.compile(r"\b(19|20)\d{2}\b")
# # # # # # # PUBDATE_RE = re.compile(r"\b(?:Publication\s*Date|Date)\s*:\s*([0-9]{4})", re.I)
# # # # # # # DOI_RE     = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
# # # # # # # URL_RE     = re.compile(r"https?://\S+")
# # # # # # # SEP_RX     = re.compile(r"\s*(?:;|,|\band\b|&|\|)\s*", re.I)

# # # # # # # driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # # # # # # # ────────────────────────────── Helpers
# # # # # # # def safe_str(x: Any) -> str:
# # # # # # #     if x is None:
# # # # # # #         return ""
# # # # # # #     if isinstance(x, float):
# # # # # # #         try:
# # # # # # #             if math.isnan(x): return ""
# # # # # # #         except Exception:
# # # # # # #             return ""
# # # # # # #     return str(x).strip()

# # # # # # # def extract_year(info: str) -> Optional[int]:
# # # # # # #     s = safe_str(info)
# # # # # # #     if not s: return None
# # # # # # #     m = PUBDATE_RE.search(s)
# # # # # # #     if m:
# # # # # # #         try: return int(m.group(1))
# # # # # # #         except: pass
# # # # # # #     m2 = DATE_RE.search(s)
# # # # # # #     return int(m2.group(0)) if m2 else None

# # # # # # # def extract_doi_link(info: str, doi: str) -> Optional[str]:
# # # # # # #     doi = safe_str(doi)
# # # # # # #     if doi:
# # # # # # #         return f"https://doi.org/{doi}" if doi.lower().startswith("10.") else doi
# # # # # # #     s = safe_str(info)
# # # # # # #     if not s: return None
# # # # # # #     m = DOI_RE.search(s)
# # # # # # #     if m: return f"https://doi.org/{m.group(0)}"
# # # # # # #     m2 = URL_RE.search(s)
# # # # # # #     return m2.group(0) if m2 else None

# # # # # # # def parse_authors(cell: Any) -> List[str]:
# # # # # # #     s = safe_str(cell)
# # # # # # #     if not s: return []
# # # # # # #     try:
# # # # # # #         if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
# # # # # # #             v = json.loads(s)
# # # # # # #             if isinstance(v, list): return _dedupe([safe_str(x) for x in v if safe_str(x)])
# # # # # # #             if isinstance(v, dict) and isinstance(v.get("authors"), list):
# # # # # # #                 return _dedupe([safe_str(x) for x in v["authors"] if safe_str(x)])
# # # # # # #     except Exception: pass
# # # # # # #     return _dedupe([p for p in SEP_RX.split(s) if p])

# # # # # # # def _dedupe(items: List[str]) -> List[str]:
# # # # # # #     seen, out = set(), []
# # # # # # #     for it in items:
# # # # # # #         k = it.lower()
# # # # # # #         if k and k not in seen:
# # # # # # #             seen.add(k)
# # # # # # #             out.append(it)
# # # # # # #     return out

# # # # # # # # ────────────────────────────── Read rows
# # # # # # # def read_rows() -> List[Dict[str, Any]]:
# # # # # # #     conn = sqlite3.connect(config.SQLITE_DB)
# # # # # # #     cur = conn.cursor()
# # # # # # #     cur.execute("PRAGMA table_info(research_info);")
# # # # # # #     cols = {r[1] for r in cur.fetchall()}
# # # # # # #     has_doi, has_pub = "doi" in cols, "publication_date" in cols

# # # # # # #     sql = "SELECT id, researcher_name, work_title, authors, info"
# # # # # # #     sql += ", doi" if has_doi else ", '' AS doi"
# # # # # # #     sql += ", publication_date" if has_pub else ", '' AS publication_date"
# # # # # # #     sql += " FROM research_info"
# # # # # # #     cur.execute(sql)
# # # # # # #     raw = cur.fetchall()
# # # # # # #     conn.close()

# # # # # # #     rows = []
# # # # # # #     for pid, rname, title, authors, info, doi, pubdate in raw:
# # # # # # #         title = safe_str(title)
# # # # # # #         if not title: continue
# # # # # # #         authors_list = parse_authors(authors)
# # # # # # #         primary = safe_str(rname) or (authors_list[0] if authors_list else "")
# # # # # # #         title_short = title[:2048]

# # # # # # #         # Embeddings
# # # # # # #         title_emb = embedder.encode(title).tolist()
# # # # # # #         primary_emb = embedder.encode(primary).tolist() if primary else None

# # # # # # #         rows.append({
# # # # # # #             "paper_id": str(pid),
# # # # # # #             "title": title,
# # # # # # #             "title_short": title_short,
# # # # # # #             "info": safe_str(info),
# # # # # # #             "primary_author": primary,
# # # # # # #             "co_authors": [a for a in authors_list if a and a.lower() != primary.lower()],
# # # # # # #             "year": extract_year(safe_str(pubdate) or safe_str(info)),
# # # # # # #             "doi_link": extract_doi_link(info, doi),
# # # # # # #             "title_emb": title_emb,
# # # # # # #             "primary_author_emb": primary_emb
# # # # # # #         })
# # # # # # #     return rows

# # # # # # # # ────────────────────────────── Schema
# # # # # # # def ensure_schema():
# # # # # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
# # # # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
# # # # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")
# # # # # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.year)")
# # # # # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.doi_link)")
# # # # # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.title_short)")

# # # # # # # # ────────────────────────────── Cypher
# # # # # # # UPSERT = """
# # # # # # # UNWIND $rows AS row
# # # # # # # MERGE (p:Paper {paper_id: row.paper_id})
# # # # # # #   SET p.title       = row.title,
# # # # # # #       p.title_short = row.title_short,
# # # # # # #       p.info        = row.info,
# # # # # # #       p.year        = row.year,
# # # # # # #       p.doi_link    = row.doi_link,
# # # # # # #       p.title_emb   = row.title_emb

# # # # # # # WITH p, row
# # # # # # # WHERE row.primary_author IS NOT NULL AND row.primary_author <> ""
# # # # # # # MERGE (r:Researcher {name: row.primary_author})
# # # # # # # MERGE (a_primary:Author {name: row.primary_author})
# # # # # # #   SET a_primary.name_emb = row.primary_author_emb
# # # # # # # MERGE (r)-[:WROTE]->(p)
# # # # # # # MERGE (a_primary)-[:AUTHORED]->(p)
# # # # # # # MERGE (p)-[:HAS_RESEARCHER]->(r)
# # # # # # # MERGE (p)-[:HAS_AUTHOR]->(a_primary)

# # # # # # # WITH p, row
# # # # # # # UNWIND row.co_authors AS coName
# # # # # # # WITH p, trim(coName) AS cname
# # # # # # # WHERE cname <> ""
# # # # # # # MERGE (a:Author {name: cname})
# # # # # # # MERGE (a)-[:AUTHORED]->(p)
# # # # # # # MERGE (p)-[:HAS_AUTHOR]->(a)
# # # # # # # """

# # # # # # # COAUTHORS = """
# # # # # # # MATCH (p:Paper)<-[:HAS_AUTHOR]-(a1:Author),
# # # # # # #       (p)<-[:HAS_AUTHOR]-(a2:Author)
# # # # # # # WHERE id(a1) < id(a2)
# # # # # # # MERGE (a1)-[:COAUTHORED_WITH]-(a2)
# # # # # # # """

# # # # # # # # ────────────────────────────── Ingest
# # # # # # # def ingest(rows: List[Dict[str, Any]]):
# # # # # # #     batch = min(getattr(config, "BATCH_SIZE", 200), 200)  # cap at 200
# # # # # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # # # # #         for i in tqdm(range(0, len(rows), batch), desc="Neo4j ingest"):
# # # # # # #             chunk = rows[i:i+batch]
# # # # # # #             s.execute_write(lambda tx: tx.run(UPSERT, rows=chunk))
# # # # # # #             print(f"  ✓ Committed {i+len(chunk)}/{len(rows)}")

# # # # # # # def build_coauthor_edges():
# # # # # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # # # # #         s.run(COAUTHORS)
# # # # # # #     print("✅ Co-author edges built.")

# # # # # # # # ────────────────────────────── Main
# # # # # # # if __name__ == "__main__":
# # # # # # #     try:
# # # # # # #         ensure_schema()
# # # # # # #     except ClientError as e:
# # # # # # #         print("Schema setup failed:", e)
# # # # # # #         raise

# # # # # # #     rows = read_rows()
# # # # # # #     print(f"📦 Rows prepared: {len(rows)}")
# # # # # # #     ingest(rows)
# # # # # # #     build_coauthor_edges()
# # # # # # #     print("✅ Done with embeddings.")


# # # # # # # neo_ingest_full.py (semantic embeddings + parallel ingestion)
# # # # # # import json, math, re, sqlite3
# # # # # # from typing import Any, Dict, List, Optional
# # # # # # from concurrent.futures import ThreadPoolExecutor, as_completed

# # # # # # from neo4j import GraphDatabase
# # # # # # from neo4j.exceptions import ClientError
# # # # # # from tqdm import tqdm
# # # # # # from sentence_transformers import SentenceTransformer

# # # # # # import config_full as config

# # # # # # # ────────────────────────────── Embedding model
# # # # # # embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # # # # # # ────────────────────────────── Regex helpers
# # # # # # DATE_RE    = re.compile(r"\b(19|20)\d{2}\b")
# # # # # # PUBDATE_RE = re.compile(r"\b(?:Publication\s*Date|Date)\s*:\s*([0-9]{4})", re.I)
# # # # # # DOI_RE     = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
# # # # # # URL_RE     = re.compile(r"https?://\S+")
# # # # # # SEP_RX     = re.compile(r"\s*(?:;|,|\band\b|&|\|)\s*", re.I)

# # # # # # driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # # # # # # ────────────────────────────── Helpers
# # # # # # def safe_str(x: Any) -> str:
# # # # # #     if x is None:
# # # # # #         return ""
# # # # # #     if isinstance(x, float):
# # # # # #         try:
# # # # # #             if math.isnan(x): return ""
# # # # # #         except Exception:
# # # # # #             return ""
# # # # # #     return str(x).strip()

# # # # # # def extract_year(info: str) -> Optional[int]:
# # # # # #     s = safe_str(info)
# # # # # #     if not s: return None
# # # # # #     m = PUBDATE_RE.search(s)
# # # # # #     if m:
# # # # # #         try: return int(m.group(1))
# # # # # #         except: pass
# # # # # #     m2 = DATE_RE.search(s)
# # # # # #     return int(m2.group(0)) if m2 else None

# # # # # # def extract_doi_link(info: str, doi: str) -> Optional[str]:
# # # # # #     doi = safe_str(doi)
# # # # # #     if doi:
# # # # # #         return f"https://doi.org/{doi}" if doi.lower().startswith("10.") else doi
# # # # # #     s = safe_str(info)
# # # # # #     if not s: return None
# # # # # #     m = DOI_RE.search(s)
# # # # # #     if m: return f"https://doi.org/{m.group(0)}"
# # # # # #     m2 = URL_RE.search(s)
# # # # # #     return m2.group(0) if m2 else None

# # # # # # def parse_authors(cell: Any) -> List[str]:
# # # # # #     s = safe_str(cell)
# # # # # #     if not s: return []
# # # # # #     try:
# # # # # #         if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
# # # # # #             v = json.loads(s)
# # # # # #             if isinstance(v, list): return _dedupe([safe_str(x) for x in v if safe_str(x)])
# # # # # #             if isinstance(v, dict) and isinstance(v.get("authors"), list):
# # # # # #                 return _dedupe([safe_str(x) for x in v["authors"] if safe_str(x)])
# # # # # #     except Exception: pass
# # # # # #     return _dedupe([p for p in SEP_RX.split(s) if p])

# # # # # # def _dedupe(items: List[str]) -> List[str]:
# # # # # #     seen, out = set(), []
# # # # # #     for it in items:
# # # # # #         k = it.lower()
# # # # # #         if k and k not in seen:
# # # # # #             seen.add(k)
# # # # # #             out.append(it)
# # # # # #     return out

# # # # # # # ────────────────────────────── Read rows
# # # # # # def read_rows() -> List[Dict[str, Any]]:
# # # # # #     conn = sqlite3.connect(config.SQLITE_DB)
# # # # # #     cur = conn.cursor()
# # # # # #     cur.execute("PRAGMA table_info(research_info);")
# # # # # #     cols = {r[1] for r in cur.fetchall()}
# # # # # #     has_doi, has_pub = "doi" in cols, "publication_date" in cols

# # # # # #     sql = "SELECT id, researcher_name, work_title, authors, info"
# # # # # #     sql += ", doi" if has_doi else ", '' AS doi"
# # # # # #     sql += ", publication_date" if has_pub else ", '' AS publication_date"
# # # # # #     sql += " FROM research_info"
# # # # # #     cur.execute(sql)
# # # # # #     raw = cur.fetchall()
# # # # # #     conn.close()

# # # # # #     rows = []
# # # # # #     for pid, rname, title, authors, info, doi, pubdate in raw:
# # # # # #         title = safe_str(title)
# # # # # #         if not title: continue
# # # # # #         authors_list = parse_authors(authors)
# # # # # #         primary = safe_str(rname) or (authors_list[0] if authors_list else "")
# # # # # #         title_short = title[:2048]

# # # # # #         # Embeddings
# # # # # #         title_emb = embedder.encode(title).tolist()
# # # # # #         primary_emb = embedder.encode(primary).tolist() if primary else None

# # # # # #         rows.append({
# # # # # #             "paper_id": str(pid),
# # # # # #             "title": title,
# # # # # #             "title_short": title_short,
# # # # # #             "info": safe_str(info),
# # # # # #             "primary_author": primary,
# # # # # #             "co_authors": [a for a in authors_list if a and a.lower() != primary.lower()],
# # # # # #             "year": extract_year(safe_str(pubdate) or safe_str(info)),
# # # # # #             "doi_link": extract_doi_link(info, doi),
# # # # # #             "title_emb": title_emb,
# # # # # #             "primary_author_emb": primary_emb
# # # # # #         })
# # # # # #     return rows

# # # # # # # ────────────────────────────── Schema
# # # # # # def ensure_schema():
# # # # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
# # # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
# # # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")
# # # # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.year)")
# # # # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.doi_link)")
# # # # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.title_short)")

# # # # # # # ────────────────────────────── Cypher
# # # # # # UPSERT = """
# # # # # # UNWIND $rows AS row
# # # # # # MERGE (p:Paper {paper_id: row.paper_id})
# # # # # #   SET p.title       = row.title,
# # # # # #       p.title_short = row.title_short,
# # # # # #       p.info        = row.info,
# # # # # #       p.year        = row.year,
# # # # # #       p.doi_link    = row.doi_link,
# # # # # #       p.title_emb   = row.title_emb

# # # # # # WITH p, row
# # # # # # WHERE row.primary_author IS NOT NULL AND row.primary_author <> ""
# # # # # # MERGE (r:Researcher {name: row.primary_author})
# # # # # # MERGE (a_primary:Author {name: row.primary_author})
# # # # # #   SET a_primary.name_emb = row.primary_author_emb
# # # # # # MERGE (r)-[:WROTE]->(p)
# # # # # # MERGE (a_primary)-[:AUTHORED]->(p)
# # # # # # MERGE (p)-[:HAS_RESEARCHER]->(r)
# # # # # # MERGE (p)-[:HAS_AUTHOR]->(a_primary)

# # # # # # WITH p, row
# # # # # # UNWIND row.co_authors AS coName
# # # # # # WITH p, trim(coName) AS cname
# # # # # # WHERE cname <> ""
# # # # # # MERGE (a:Author {name: cname})
# # # # # # MERGE (a)-[:AUTHORED]->(p)
# # # # # # MERGE (p)-[:HAS_AUTHOR]->(a)
# # # # # # """

# # # # # # COAUTHORS = """
# # # # # # MATCH (p:Paper)<-[:HAS_AUTHOR]-(a1:Author),
# # # # # #       (p)<-[:HAS_AUTHOR]-(a2:Author)
# # # # # # WHERE id(a1) < id(a2)
# # # # # # MERGE (a1)-[:COAUTHORED_WITH]-(a2)
# # # # # # """

# # # # # # # ────────────────────────────── Parallel Ingest
# # # # # # def ingest(rows: List[Dict[str, Any]]):
# # # # # #     batch_size = min(getattr(config, "BATCH_SIZE", 200), 200)
# # # # # #     batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]

# # # # # #     def worker(chunk):
# # # # # #         with driver.session(database=config.NEO4J_DB) as s:
# # # # # #             s.execute_write(lambda tx: tx.run(UPSERT, rows=chunk))
# # # # # #         return len(chunk)

# # # # # #     done, total = 0, len(rows)
# # # # # #     with ThreadPoolExecutor(max_workers=8) as executor:  # tuned for i9-185H
# # # # # #         futures = [executor.submit(worker, b) for b in batches]
# # # # # #         for f in as_completed(futures):
# # # # # #             committed = f.result()
# # # # # #             done += committed
# # # # # #             print(f"  ✓ Committed {done}/{total} rows")

# # # # # # def build_coauthor_edges():
# # # # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # # # #         s.run(COAUTHORS)
# # # # # #     print("✅ Co-author edges built.")

# # # # # # # ────────────────────────────── Main
# # # # # # if __name__ == "__main__":
# # # # # #     try:
# # # # # #         ensure_schema()
# # # # # #     except ClientError as e:
# # # # # #         print("Schema setup failed:", e)
# # # # # #         raise

# # # # # #     rows = read_rows()
# # # # # #     print(f"📦 Rows prepared: {len(rows)}")
# # # # # #     ingest(rows)
# # # # # #     build_coauthor_edges()
# # # # # #     print("✅ Done with embeddings + parallel ingest.")

# # # # # # neo_ingest_full.py (with cached embeddings + progress logging)
# # # # # import json, math, re, sqlite3, os
# # # # # from typing import Any, Dict, List, Optional
# # # # # from concurrent.futures import ThreadPoolExecutor, as_completed

# # # # # from neo4j import GraphDatabase
# # # # # from neo4j.exceptions import ClientError
# # # # # from tqdm import tqdm
# # # # # from sentence_transformers import SentenceTransformer

# # # # # import config_full as config

# # # # # # ────────────────────────────── Hugging Face cache dir
# # # # # HF_CACHE = os.path.join("D:/OSPO/hf_models")  # you can change if needed
# # # # # os.makedirs(HF_CACHE, exist_ok=True)

# # # # # # ────────────────────────────── Embedding model (cached)
# # # # # print("🔄 Loading embedding model (MiniLM-L6-v2)...")
# # # # # embedder = SentenceTransformer(
# # # # #     "sentence-transformers/all-MiniLM-L6-v2",
# # # # #     cache_folder=HF_CACHE
# # # # # )
# # # # # print("✅ Embedding model loaded.")

# # # # # # ────────────────────────────── Regex helpers
# # # # # DATE_RE    = re.compile(r"\b(19|20)\d{2}\b")
# # # # # PUBDATE_RE = re.compile(r"\b(?:Publication\s*Date|Date)\s*:\s*([0-9]{4})", re.I)
# # # # # DOI_RE     = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
# # # # # URL_RE     = re.compile(r"https?://\S+")
# # # # # SEP_RX     = re.compile(r"\s*(?:;|,|\band\b|&|\|)\s*", re.I)

# # # # # driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # # # # # ────────────────────────────── Helpers
# # # # # def safe_str(x: Any) -> str:
# # # # #     if x is None:
# # # # #         return ""
# # # # #     if isinstance(x, float):
# # # # #         try:
# # # # #             if math.isnan(x): return ""
# # # # #         except Exception:
# # # # #             return ""
# # # # #     return str(x).strip()

# # # # # def extract_year(info: str) -> Optional[int]:
# # # # #     s = safe_str(info)
# # # # #     if not s: return None
# # # # #     m = PUBDATE_RE.search(s)
# # # # #     if m:
# # # # #         try: return int(m.group(1))
# # # # #         except: pass
# # # # #     m2 = DATE_RE.search(s)
# # # # #     return int(m2.group(0)) if m2 else None

# # # # # def extract_doi_link(info: str, doi: str) -> Optional[str]:
# # # # #     doi = safe_str(doi)
# # # # #     if doi:
# # # # #         return f"https://doi.org/{doi}" if doi.lower().startswith("10.") else doi
# # # # #     s = safe_str(info)
# # # # #     if not s: return None
# # # # #     m = DOI_RE.search(s)
# # # # #     if m: return f"https://doi.org/{m.group(0)}"
# # # # #     m2 = URL_RE.search(s)
# # # # #     return m2.group(0) if m2 else None

# # # # # def parse_authors(cell: Any) -> List[str]:
# # # # #     s = safe_str(cell)
# # # # #     if not s: return []
# # # # #     try:
# # # # #         if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
# # # # #             v = json.loads(s)
# # # # #             if isinstance(v, list): return _dedupe([safe_str(x) for x in v if safe_str(x)])
# # # # #             if isinstance(v, dict) and isinstance(v.get("authors"), list):
# # # # #                 return _dedupe([safe_str(x) for x in v["authors"] if safe_str(x)])
# # # # #     except Exception: pass
# # # # #     return _dedupe([p for p in SEP_RX.split(s) if p])

# # # # # def _dedupe(items: List[str]) -> List[str]:
# # # # #     seen, out = set(), []
# # # # #     for it in items:
# # # # #         k = it.lower()
# # # # #         if k and k not in seen:
# # # # #             seen.add(k)
# # # # #             out.append(it)
# # # # #     return out

# # # # # # ────────────────────────────── Read rows
# # # # # def read_rows() -> List[Dict[str, Any]]:
# # # # #     conn = sqlite3.connect(config.SQLITE_DB)
# # # # #     cur = conn.cursor()
# # # # #     cur.execute("PRAGMA table_info(research_info);")
# # # # #     cols = {r[1] for r in cur.fetchall()}
# # # # #     has_doi, has_pub = "doi" in cols, "publication_date" in cols

# # # # #     sql = "SELECT id, researcher_name, work_title, authors, info"
# # # # #     sql += ", doi" if has_doi else ", '' AS doi"
# # # # #     sql += ", publication_date" if has_pub else ", '' AS publication_date"
# # # # #     sql += " FROM research_info"
# # # # #     cur.execute(sql)
# # # # #     raw = cur.fetchall()
# # # # #     conn.close()

# # # # #     rows = []
# # # # #     for idx, (pid, rname, title, authors, info, doi, pubdate) in enumerate(raw, 1):
# # # # #         title = safe_str(title)
# # # # #         if not title: continue
# # # # #         authors_list = parse_authors(authors)
# # # # #         primary = safe_str(rname) or (authors_list[0] if authors_list else "")
# # # # #         title_short = title[:2048]

# # # # #         # Embeddings with progress log
# # # # #         if idx % 100 == 0:
# # # # #             print(f"🔄 Embedding row {idx}/{len(raw)}: {title[:60]}...")

# # # # #         title_emb = embedder.encode(title).tolist()
# # # # #         primary_emb = embedder.encode(primary).tolist() if primary else None

# # # # #         rows.append({
# # # # #             "paper_id": str(pid),
# # # # #             "title": title,
# # # # #             "title_short": title_short,
# # # # #             "info": safe_str(info),
# # # # #             "primary_author": primary,
# # # # #             "co_authors": [a for a in authors_list if a and a.lower() != primary.lower()],
# # # # #             "year": extract_year(safe_str(pubdate) or safe_str(info)),
# # # # #             "doi_link": extract_doi_link(info, doi),
# # # # #             "title_emb": title_emb,
# # # # #             "primary_author_emb": primary_emb
# # # # #         })
# # # # #     return rows

# # # # # # ────────────────────────────── Schema
# # # # # def ensure_schema():
# # # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
# # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
# # # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")
# # # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.year)")
# # # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.doi_link)")
# # # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.title_short)")

# # # # # # ────────────────────────────── Cypher
# # # # # UPSERT = """
# # # # # UNWIND $rows AS row
# # # # # MERGE (p:Paper {paper_id: row.paper_id})
# # # # #   SET p.title       = row.title,
# # # # #       p.title_short = row.title_short,
# # # # #       p.info        = row.info,
# # # # #       p.year        = row.year,
# # # # #       p.doi_link    = row.doi_link,
# # # # #       p.title_emb   = row.title_emb

# # # # # WITH p, row
# # # # # WHERE row.primary_author IS NOT NULL AND row.primary_author <> ""
# # # # # MERGE (r:Researcher {name: row.primary_author})
# # # # # MERGE (a_primary:Author {name: row.primary_author})
# # # # #   SET a_primary.name_emb = row.primary_author_emb
# # # # # MERGE (r)-[:WROTE]->(p)
# # # # # MERGE (a_primary)-[:AUTHORED]->(p)
# # # # # MERGE (p)-[:HAS_RESEARCHER]->(r)
# # # # # MERGE (p)-[:HAS_AUTHOR]->(a_primary)

# # # # # WITH p, row
# # # # # UNWIND row.co_authors AS coName
# # # # # WITH p, trim(coName) AS cname
# # # # # WHERE cname <> ""
# # # # # MERGE (a:Author {name: cname})
# # # # # MERGE (a)-[:AUTHORED]->(p)
# # # # # MERGE (p)-[:HAS_AUTHOR]->(a)
# # # # # """

# # # # # COAUTHORS = """
# # # # # MATCH (p:Paper)<-[:HAS_AUTHOR]-(a1:Author),
# # # # #       (p)<-[:HAS_AUTHOR]-(a2:Author)
# # # # # WHERE id(a1) < id(a2)
# # # # # MERGE (a1)-[:COAUTHORED_WITH]-(a2)
# # # # # """

# # # # # # ────────────────────────────── Parallel Ingest
# # # # # def ingest(rows: List[Dict[str, Any]]):
# # # # #     batch_size = min(getattr(config, "BATCH_SIZE", 200), 200)
# # # # #     batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]

# # # # #     def worker(chunk):
# # # # #         with driver.session(database=config.NEO4J_DB) as s:
# # # # #             s.execute_write(lambda tx: tx.run(UPSERT, rows=chunk))
# # # # #         return len(chunk)

# # # # #     done, total = 0, len(rows)
# # # # #     with ThreadPoolExecutor(max_workers=8) as executor:  # tuned for i9-185H
# # # # #         futures = [executor.submit(worker, b) for b in batches]
# # # # #         for f in as_completed(futures):
# # # # #             committed = f.result()
# # # # #             done += committed
# # # # #             print(f"  ✓ Committed {done}/{total} rows")

# # # # # def build_coauthor_edges():
# # # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # # #         s.run(COAUTHORS)
# # # # #     print("✅ Co-author edges built.")

# # # # # # ────────────────────────────── Main
# # # # # if __name__ == "__main__":
# # # # #     try:
# # # # #         ensure_schema()
# # # # #     except ClientError as e:
# # # # #         print("Schema setup failed:", e)
# # # # #         raise

# # # # #     rows = read_rows()
# # # # #     print(f"📦 Rows prepared: {len(rows)}")
# # # # #     ingest(rows)
# # # # #     build_coauthor_edges()
# # # # #     print("✅ Done with embeddings + parallel ingest (cached model).")


# # # # # neo_ingest_full.py (with cached embeddings + progress logging + vector indexes)
# # # # import json, math, re, sqlite3, os
# # # # from typing import Any, Dict, List, Optional
# # # # from concurrent.futures import ThreadPoolExecutor, as_completed

# # # # from neo4j import GraphDatabase
# # # # from neo4j.exceptions import ClientError
# # # # from sentence_transformers import SentenceTransformer
# # # # import config_full as config

# # # # # Hugging Face cache dir (optional)
# # # # HF_CACHE = os.path.join("D:/OSPO/hf_models")
# # # # os.makedirs(HF_CACHE, exist_ok=True)

# # # # print("Loading embedding model (MiniLM-L6-v2)...")
# # # # embedder = SentenceTransformer(
# # # #     "sentence-transformers/all-MiniLM-L6-v2",
# # # #     cache_folder=HF_CACHE
# # # # )
# # # # print("Embedding model loaded.")

# # # # DATE_RE    = re.compile(r"\b(19|20)\d{2}\b")
# # # # PUBDATE_RE = re.compile(r"\b(?:Publication\s*Date|Date)\s*:\s*([0-9]{4})", re.I)
# # # # DOI_RE     = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
# # # # URL_RE     = re.compile(r"https?://\S+")
# # # # SEP_RX     = re.compile(r"\s*(?:;|,|\band\b|&|\|)\s*", re.I)

# # # # driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # # # def safe_str(x: Any) -> str:
# # # #     if x is None: return ""
# # # #     if isinstance(x, float):
# # # #         try:
# # # #             if math.isnan(x): return ""
# # # #         except Exception:
# # # #             return ""
# # # #     return str(x).strip()

# # # # def extract_year(info: str) -> Optional[int]:
# # # #     s = safe_str(info)
# # # #     if not s: return None
# # # #     m = PUBDATE_RE.search(s)
# # # #     if m:
# # # #         try: return int(m.group(1))
# # # #         except: pass
# # # #     m2 = DATE_RE.search(s)
# # # #     return int(m2.group(0)) if m2 else None

# # # # def extract_doi_link(info: str, doi: str) -> Optional[str]:
# # # #     doi = safe_str(doi)
# # # #     if doi:
# # # #         return f"https://doi.org/{doi}" if doi.lower().startswith("10.") else doi
# # # #     s = safe_str(info)
# # # #     if not s: return None
# # # #     m = DOI_RE.search(s)
# # # #     if m: return f"https://doi.org/{m.group(0)}"
# # # #     m2 = URL_RE.search(s)
# # # #     return m2.group(0) if m2 else None

# # # # def _dedupe(items: List[str]) -> List[str]:
# # # #     seen, out = set(), []
# # # #     for it in items:
# # # #         k = it.lower()
# # # #         if k and k not in seen:
# # # #             seen.add(k)
# # # #             out.append(it)
# # # #     return out

# # # # def parse_authors(cell: Any) -> List[str]:
# # # #     s = safe_str(cell)
# # # #     if not s: return []
# # # #     try:
# # # #         if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
# # # #             v = json.loads(s)
# # # #             if isinstance(v, list): return _dedupe([safe_str(x) for x in v if safe_str(x)])
# # # #             if isinstance(v, dict) and isinstance(v.get("authors"), list):
# # # #                 return _dedupe([safe_str(x) for x in v["authors"] if safe_str(x)])
# # # #     except Exception:
# # # #         pass
# # # #     return _dedupe([p for p in SEP_RX.split(s) if p])

# # # # def read_rows() -> List[Dict[str, Any]]:
# # # #     conn = sqlite3.connect(config.SQLITE_DB)
# # # #     cur = conn.cursor()
# # # #     cur.execute("PRAGMA table_info(research_info);")
# # # #     cols = {r[1] for r in cur.fetchall()}
# # # #     has_doi, has_pub = "doi" in cols, "publication_date" in cols

# # # #     sql = "SELECT id, researcher_name, work_title, authors, info"
# # # #     sql += ", doi" if has_doi else ", '' AS doi"
# # # #     sql += ", publication_date" if has_pub else ", '' AS publication_date"
# # # #     sql += " FROM research_info"
# # # #     cur.execute(sql)
# # # #     raw = cur.fetchall()
# # # #     conn.close()

# # # #     rows = []
# # # #     for idx, (pid, rname, title, authors, info, doi, pubdate) in enumerate(raw, 1):
# # # #         title = safe_str(title)
# # # #         if not title: continue
# # # #         authors_list = parse_authors(authors)
# # # #         primary = safe_str(rname) or (authors_list[0] if authors_list else "")
# # # #         title_short = title[:2048]

# # # #         if idx % 100 == 0:
# # # #             print(f"Embedding row {idx}/{len(raw)}: {title[:60]}...")

# # # #         title_emb = embedder.encode(title).tolist()
# # # #         primary_emb = embedder.encode(primary).tolist() if primary else None

# # # #         rows.append({
# # # #             "paper_id": str(pid),
# # # #             "title": title,
# # # #             "title_short": title_short,
# # # #             "info": safe_str(info),
# # # #             "primary_author": primary,
# # # #             "co_authors": [a for a in authors_list if a and a.lower() != primary.lower()],
# # # #             "year": extract_year(safe_str(pubdate) or safe_str(info)),
# # # #             "doi_link": extract_doi_link(info, doi),
# # # #             "title_emb": title_emb,
# # # #             "primary_author_emb": primary_emb
# # # #         })
# # # #     return rows

# # # # def ensure_schema():
# # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
# # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
# # # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")
# # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.year)")
# # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.doi_link)")
# # # #         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.title_short)")

# # # # def ensure_vector_indexes():
# # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # #         s.run("""
# # # #         CREATE VECTOR INDEX IF NOT EXISTS paper_title_index
# # # #         FOR (p:Paper) ON (p.title_emb)
# # # #         OPTIONS { indexConfig: { 'vector.dimensions': 384, 'vector.similarity_function': 'cosine' } }
# # # #         """)
# # # #         s.run("""
# # # #         CREATE VECTOR INDEX IF NOT EXISTS author_name_index
# # # #         FOR (a:Author) ON (a.name_emb)
# # # #         OPTIONS { indexConfig: { 'vector.dimensions': 384, 'vector.similarity_function': 'cosine' } }
# # # #         """)
# # # #     print("Vector indexes ensured.")

# # # # UPSERT = """
# # # # UNWIND $rows AS row
# # # # MERGE (p:Paper {paper_id: row.paper_id})
# # # #   SET p.title       = row.title,
# # # #       p.title_short = row.title_short,
# # # #       p.info        = row.info,
# # # #       p.year        = row.year,
# # # #       p.doi_link    = row.doi_link,
# # # #       p.title_emb   = row.title_emb

# # # # WITH p, row
# # # # WHERE row.primary_author IS NOT NULL AND row.primary_author <> ""
# # # # MERGE (r:Researcher {name: row.primary_author})
# # # # MERGE (a_primary:Author {name: row.primary_author})
# # # # FOREACH (_ IN CASE WHEN row.primary_author_emb IS NULL THEN [] ELSE [1] END |
# # # #   SET a_primary.name_emb = row.primary_author_emb
# # # # )
# # # # MERGE (r)-[:WROTE]->(p)
# # # # MERGE (a_primary)-[:AUTHORED]->(p)
# # # # MERGE (p)-[:HAS_RESEARCHER]->(r)
# # # # MERGE (p)-[:HAS_AUTHOR]->(a_primary)

# # # # WITH p, row
# # # # UNWIND row.co_authors AS coName
# # # # WITH p, trim(coName) AS cname
# # # # WHERE cname <> ""
# # # # MERGE (a:Author {name: cname})
# # # # MERGE (a)-[:AUTHORED]->(p)
# # # # MERGE (p)-[:HAS_AUTHOR]->(a)
# # # # """

# # # # COAUTHORS = """
# # # # MATCH (p:Paper)<-[:HAS_AUTHOR]-(a1:Author),
# # # #       (p)<-[:HAS_AUTHOR]-(a2:Author)
# # # # WHERE id(a1) < id(a2)
# # # # MERGE (a1)-[:COAUTHORED_WITH]-(a2)
# # # # """

# # # # def ingest(rows: List[Dict[str, Any]]):
# # # #     batch_size = min(getattr(config, "BATCH_SIZE", 200), 200)
# # # #     batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]

# # # #     def worker(chunk):
# # # #         with driver.session(database=config.NEO4J_DB) as s:
# # # #             s.execute_write(lambda tx: tx.run(UPSERT, rows=chunk))
# # # #         return len(chunk)

# # # #     done, total = 0, len(rows)
# # # #     with ThreadPoolExecutor(max_workers=8) as executor:
# # # #         futures = [executor.submit(worker, b) for b in batches]
# # # #         for f in as_completed(futures):
# # # #             done += f.result()
# # # #             print(f"Committed {done}/{total} rows")

# # # # def build_coauthor_edges():
# # # #     with driver.session(database=config.NEO4J_DB) as s:
# # # #         s.run(COAUTHORS)
# # # #     print("Co-author edges built.")

# # # # if __name__ == "__main__":
# # # #     try:
# # # #         ensure_schema()
# # # #         ensure_vector_indexes()
# # # #     except ClientError as e:
# # # #         print("Schema/index setup failed:", e)
# # # #         raise

# # # #     rows = read_rows()
# # # #     print(f"Rows prepared: {len(rows)}")
# # # #     ingest(rows)
# # # #     build_coauthor_edges()
# # # #     print("Done with embeddings + parallel ingest (cached model).")

# # # # neo_ingest_full.py 
# # # import json, math, re, sqlite3, os
# # # from typing import Any, Dict, List, Optional
# # # from concurrent.futures import ThreadPoolExecutor, as_completed

# # # from neo4j import GraphDatabase
# # # from neo4j.exceptions import ClientError
# # # from sentence_transformers import SentenceTransformer
# # # import config_full as config

# # # HF_CACHE = os.path.join("D:/OSPO/hf_models")
# # # os.makedirs(HF_CACHE, exist_ok=True)

# # # print("Loading embedding model (MiniLM-L6-v2)...")
# # # embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=HF_CACHE)
# # # print("Embedding model loaded.")

# # # DATE_RE    = re.compile(r"\b(19|20)\d{2}\b")
# # # PUBDATE_RE = re.compile(r"\b(?:Publication\s*Date|Date)\s*:\s*([0-9]{4})", re.I)
# # # DOI_RE     = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
# # # URL_RE     = re.compile(r"https?://\S+")
# # # SEP_RX     = re.compile(r"\s*(?:;|,|\band\b|&|\|)\s*", re.I)

# # # driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # # def safe_str(x: Any) -> str:
# # #     if x is None: return ""
# # #     if isinstance(x, float):
# # #         try:
# # #             if math.isnan(x): return ""
# # #         except Exception:
# # #             return ""
# # #     return str(x).strip()

# # # def extract_year(info: str) -> Optional[int]:
# # #     s = safe_str(info)
# # #     if not s: return None
# # #     m = PUBDATE_RE.search(s)
# # #     if m:
# # #         try: return int(m.group(1))
# # #         except: pass
# # #     m2 = DATE_RE.search(s)
# # #     return int(m2.group(0)) if m2 else None

# # # def extract_doi_link(info: str, doi: str) -> Optional[str]:
# # #     doi = safe_str(doi)
# # #     if doi:
# # #         return f"https://doi.org/{doi}" if doi.lower().startswith("10.") else doi
# # #     s = safe_str(info)
# # #     if not s: return None
# # #     m = DOI_RE.search(s)
# # #     if m: return f"https://doi.org/{m.group(0)}"
# # #     m2 = URL_RE.search(s)
# # #     return m2.group(0) if m2 else None

# # # def _dedupe(items: List[str]) -> List[str]:
# # #     seen, out = set(), []
# # #     for it in items:
# # #         k = it.lower()
# # #         if k and k not in seen:
# # #             seen.add(k)
# # #             out.append(it)
# # #     return out

# # # def parse_authors(cell: Any) -> List[str]:
# # #     s = safe_str(cell)
# # #     if not s: return []
# # #     try:
# # #         if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
# # #             v = json.loads(s)
# # #             if isinstance(v, list): return _dedupe([safe_str(x) for x in v if safe_str(x)])
# # #             if isinstance(v, dict) and isinstance(v.get("authors"), list):
# # #                 return _dedupe([safe_str(x) for x in v["authors"] if safe_str(x)])
# # #     except Exception:
# # #         pass
# # #     return _dedupe([p for p in SEP_RX.split(s) if p])

# # # def read_rows() -> List[Dict[str, Any]]:
# # #     conn = sqlite3.connect(config.SQLITE_DB)
# # #     cur = conn.cursor()
# # #     cur.execute("PRAGMA table_info(research_info);")
# # #     cols = {r[1] for r in cur.fetchall()}
# # #     has_doi, has_pub = "doi" in cols, "publication_date" in cols

# # #     sql = "SELECT id, researcher_name, work_title, authors, info"
# # #     sql += ", doi" if has_doi else ", '' AS doi"
# # #     sql += ", publication_date" if has_pub else ", '' AS publication_date"
# # #     sql += " FROM research_info"
# # #     cur.execute(sql)
# # #     raw = cur.fetchall()
# # #     conn.close()

# # #     rows = []
# # #     for idx, (pid, rname, title, authors, info, doi, pubdate) in enumerate(raw, 1):
# # #         title = safe_str(title)
# # #         if not title: continue
# # #         authors_list = parse_authors(authors)
# # #         primary = safe_str(rname) or (authors_list[0] if authors_list else "")
# # #         title_short = title[:2048]

# # #         rows.append({
# # #             "paper_id": str(pid),
# # #             "title": title,
# # #             "title_short": title_short,
# # #             "info": safe_str(info),
# # #             "primary_author": primary,
# # #             "co_authors": [a for a in authors_list if a and a.lower() != primary.lower()],
# # #             "year": extract_year(safe_str(pubdate) or safe_str(info)),
# # #             "doi_link": extract_doi_link(info, doi)
# # #         })
# # #     return rows

# # # def ensure_schema():
# # #     with driver.session(database=config.NEO4J_DB) as s:
# # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
# # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
# # #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")

# # # UPSERT = """
# # # UNWIND $rows AS row
# # # MERGE (p:Paper {paper_id: row.paper_id})
# # #   SET p.title       = row.title,
# # #       p.title_short = row.title_short,
# # #       p.info        = row.info,
# # #       p.year        = row.year,
# # #       p.doi_link    = row.doi_link

# # # WITH p, row
# # # WHERE row.primary_author IS NOT NULL AND row.primary_author <> ""
# # # MERGE (r:Researcher {name: row.primary_author})
# # # MERGE (a_primary:Author {name: row.primary_author})
# # # MERGE (r)-[:WROTE]->(p)
# # # MERGE (a_primary)-[:AUTHORED]->(p)
# # # MERGE (p)-[:HAS_RESEARCHER]->(r)
# # # MERGE (p)-[:HAS_AUTHOR]->(a_primary)

# # # WITH p, row
# # # UNWIND row.co_authors AS coName
# # # WITH p, trim(coName) AS cname
# # # WHERE cname <> ""
# # # MERGE (a:Author {name: cname})
# # # MERGE (a)-[:AUTHORED]->(p)
# # # MERGE (p)-[:HAS_AUTHOR]->(a)
# # # """

# # # COAUTHORS = """
# # # MATCH (p:Paper)<-[:HAS_AUTHOR]-(a1:Author),
# # #       (p)<-[:HAS_AUTHOR]-(a2:Author)
# # # WHERE id(a1) < id(a2)
# # # MERGE (a1)-[:COAUTHORED_WITH]-(a2)
# # # """

# # # def ingest(rows: List[Dict[str, Any]]):
# # #     batch_size = 200
# # #     batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]

# # #     def worker(chunk):
# # #         with driver.session(database=config.NEO4J_DB) as s:
# # #             s.execute_write(lambda tx: tx.run(UPSERT, rows=chunk))
# # #         return len(chunk)

# # #     done, total = 0, len(rows)
# # #     with ThreadPoolExecutor(max_workers=8) as executor:
# # #         futures = [executor.submit(worker, b) for b in batches]
# # #         for f in as_completed(futures):
# # #             done += f.result()
# # #             print(f"Committed {done}/{total} rows")

# # # def build_coauthor_edges():
# # #     with driver.session(database=config.NEO4J_DB) as s:
# # #         s.run(COAUTHORS)
# # #     print("Co-author edges built.")

# # # if __name__ == "__main__":
# # #     try:
# # #         ensure_schema()
# # #     except ClientError as e:
# # #         print("Schema setup failed:", e)
# # #         raise

# # #     rows = read_rows()
# # #     print(f"Rows prepared: {len(rows)}")
# # #     ingest(rows)
# # #     build_coauthor_edges()
# # #     print("✅ Done ingesting into Neo4j CE.")


# # import json, math, re, sqlite3, os
# # from typing import Any, Dict, List, Optional
# # from concurrent.futures import ThreadPoolExecutor, as_completed

# # from neo4j import GraphDatabase
# # from neo4j.exceptions import ClientError
# # from sentence_transformers import SentenceTransformer
# # import config_full as config

# # HF_CACHE = os.path.join("D:/OSPO/hf_models")
# # os.makedirs(HF_CACHE, exist_ok=True)

# # print("Loading embedding model (MiniLM-L6-v2)...")
# # embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=HF_CACHE)
# # print("Embedding model loaded.")

# # DATE_RE    = re.compile(r"\b(19|20)\d{2}\b")
# # PUBDATE_RE = re.compile(r"\b(?:Publication\s*Date|Date)\s*:\s*([0-9]{4})", re.I)
# # DOI_RE     = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
# # URL_RE     = re.compile(r"https?://\S+")
# # SEP_RX     = re.compile(r"\s*(?:;|,|\band\b|&|\|)\s*", re.I)

# # driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # def safe_str(x: Any) -> str:
# #     if x is None: return ""
# #     if isinstance(x, float):
# #         try:
# #             if math.isnan(x): return ""
# #         except Exception:
# #             return ""
# #     return str(x).strip()

# # def extract_year(info: str) -> Optional[int]:
# #     s = safe_str(info)
# #     if not s: return None
# #     m = PUBDATE_RE.search(s)
# #     if m:
# #         try: return int(m.group(1))
# #         except: pass
# #     m2 = DATE_RE.search(s)
# #     return int(m2.group(0)) if m2 else None

# # def extract_doi_link(info: str, doi: str) -> Optional[str]:
# #     doi = safe_str(doi)
# #     if doi:
# #         return f"https://doi.org/{doi}" if doi.lower().startswith("10.") else doi
# #     s = safe_str(info)
# #     if not s: return None
# #     m = DOI_RE.search(s)
# #     if m: return f"https://doi.org/{m.group(0)}"
# #     m2 = URL_RE.search(s)
# #     return m2.group(0) if m2 else None

# # def _dedupe(items: List[str]) -> List[str]:
# #     seen, out = set(), []
# #     for it in items:
# #         k = it.lower()
# #         if k and k not in seen:
# #             seen.add(k)
# #             out.append(it)
# #     return out

# # def parse_authors(cell: Any) -> List[str]:
# #     s = safe_str(cell)
# #     if not s: return []
# #     try:
# #         if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
# #             v = json.loads(s)
# #             if isinstance(v, list): return _dedupe([safe_str(x) for x in v if safe_str(x)])
# #             if isinstance(v, dict) and isinstance(v.get("authors"), list):
# #                 return _dedupe([safe_str(x) for x in v["authors"] if safe_str(x)])
# #     except Exception:
# #         pass
# #     return _dedupe([p for p in SEP_RX.split(s) if p])

# # def read_rows() -> List[Dict[str, Any]]:
# #     conn = sqlite3.connect(config.SQLITE_DB)
# #     cur = conn.cursor()
# #     cur.execute("PRAGMA table_info(research_info);")
# #     cols = {r[1] for r in cur.fetchall()}
# #     has_doi, has_pub = "doi" in cols, "publication_date" in cols

# #     sql = "SELECT id, researcher_name, work_title, authors, info"
# #     sql += ", doi" if has_doi else ", '' AS doi"
# #     sql += ", publication_date" if has_pub else ", '' AS publication_date"
# #     sql += " FROM research_info"
# #     cur.execute(sql)
# #     raw = cur.fetchall()
# #     conn.close()

# #     rows = []
# #     for idx, (pid, rname, title, authors, info, doi, pubdate) in enumerate(raw, 1):
# #         title = safe_str(title)
# #         if not title: continue
# #         authors_list = parse_authors(authors)
# #         primary = safe_str(rname) or (authors_list[0] if authors_list else "")
# #         title_short = title[:2048]

# #         rows.append({
# #             "paper_id": str(pid),
# #             "title": title,
# #             "title_short": title_short,
# #             "info": safe_str(info),
# #             "primary_author": primary,
# #             "co_authors": [a for a in authors_list if a and a.lower() != primary.lower()],
# #             "year": extract_year(safe_str(pubdate) or safe_str(info)),
# #             "doi_link": extract_doi_link(info, doi)
# #         })
# #     return rows

# # def ensure_schema():
# #     with driver.session(database=config.NEO4J_DB) as s:
# #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
# #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
# #         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")

# # UPSERT = """
# # UNWIND $rows AS row
# # MERGE (p:Paper {paper_id: row.paper_id})
# #   SET p.title       = row.title,
# #       p.title_short = row.title_short,
# #       p.info        = row.info,
# #       p.year        = row.year,
# #       p.doi_link    = row.doi_link

# # WITH p, row
# # WHERE row.primary_author IS NOT NULL AND row.primary_author <> ""
# # MERGE (r:Researcher {name: row.primary_author})
# # MERGE (a_primary:Author {name: row.primary_author})
# # MERGE (r)-[:WROTE]->(p)
# # MERGE (a_primary)-[:AUTHORED]->(p)
# # MERGE (p)-[:HAS_RESEARCHER]->(r)
# # MERGE (p)-[:HAS_AUTHOR]->(a_primary)

# # WITH p, row
# # UNWIND row.co_authors AS coName
# # WITH p, trim(coName) AS cname
# # WHERE cname <> ""
# # MERGE (a:Author {name: cname})
# # MERGE (a)-[:AUTHORED]->(p)
# # MERGE (p)-[:HAS_AUTHOR]->(a)
# # """

# # COAUTHORS = """
# # MATCH (p:Paper)<-[:HAS_AUTHOR]-(a1:Author),
# #       (p)<-[:HAS_AUTHOR]-(a2:Author)
# # WHERE id(a1) < id(a2)
# # MERGE (a1)-[:COAUTHORED_WITH]-(a2)
# # """

# # def ingest(rows: List[Dict[str, Any]]):
# #     batch_size = 200
# #     batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]

# #     def worker(chunk):
# #         with driver.session(database=config.NEO4J_DB) as s:
# #             s.execute_write(lambda tx: tx.run(UPSERT, rows=chunk))
# #         return len(chunk)

# #     done, total = 0, len(rows)
# #     with ThreadPoolExecutor(max_workers=8) as executor:
# #         futures = [executor.submit(worker, b) for b in batches]
# #         for f in as_completed(futures):
# #             done += f.result()
# #             print(f"Committed {done}/{total} rows")

# # def build_coauthor_edges():
# #     with driver.session(database=config.NEO4J_DB) as s:
# #         s.run(COAUTHORS)
# #     print("Co-author edges built.")

# # if __name__ == "__main__":
# #     try:
# #         ensure_schema()
# #     except ClientError as e:
# #         print("Schema setup failed:", e)
# #         raise

# #     rows = read_rows()
# #     print(f"Rows prepared: {len(rows)}")
# #     ingest(rows)
# #     build_coauthor_edges()
# #     print("✅ Done ingesting into Neo4j CE.")


# # neo_ingest_full.py
# """
# Neo4j Community Edition ingest (no vector indexes).
# - Pulls from SQLite (config.SQLITE_DB) and builds a simple scholarly KG:
#   (:Paper)-[:HAS_AUTHOR]->(:Author)
#   (:Researcher)-[:WROTE]->(:Paper)
#   plus symmetric COAUTHORED_WITH edges between Authors who share a Paper.
# - Idempotent (MERGE everywhere), batched, and with basic cleaning.

# Requirements:
# - Neo4j CE running locally (default bolt://localhost:7687)
# - Credentials from config_full.py (env-overridable)
# """

# import json
# import math
# import os
# import re
# import sqlite3
# from typing import Any, Dict, List, Optional
# from concurrent.futures import ThreadPoolExecutor, as_completed

# from neo4j import GraphDatabase
# from neo4j.exceptions import ClientError

# import config_full as config

# # --------------------------- helpers / cleaning -------------------------------

# DATE_RE    = re.compile(r"\b(19|20)\d{2}\b")
# PUBDATE_RE = re.compile(r"\b(?:Publication\s*Date|Date)\s*:\s*([0-9]{4})", re.I)
# DOI_RE     = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
# URL_RE     = re.compile(r"https?://\S+")
# SEP_RX     = re.compile(r"\s*(?:;|,|\band\b|&|\|)\s*", re.I)

# def safe_str(x: Any) -> str:
#     if x is None:
#         return ""
#     if isinstance(x, float):
#         try:
#             if math.isnan(x):
#                 return ""
#         except Exception:
#             return ""
#     return str(x).strip()

# def extract_year(info: str, pubdate: str = "") -> Optional[int]:
#     s = safe_str(pubdate) or safe_str(info)
#     if not s:
#         return None
#     m = PUBDATE_RE.search(s)
#     if m:
#         try:
#             return int(m.group(1))
#         except Exception:
#             pass
#     m2 = DATE_RE.search(s)
#     return int(m2.group(0)) if m2 else None

# def extract_doi_link(info: str, doi: str) -> Optional[str]:
#     doi = safe_str(doi)
#     if doi:
#         return f"https://doi.org/{doi}" if doi.lower().startswith("10.") else doi
#     s = safe_str(info)
#     if not s:
#         return None
#     m = DOI_RE.search(s)
#     if m:
#         return f"https://doi.org/{m.group(0)}"
#     m2 = URL_RE.search(s)
#     return m2.group(0) if m2 else None

# def _dedupe(items: List[str]) -> List[str]:
#     seen, out = set(), []
#     for it in items:
#         k = it.lower().strip()
#         if k and k not in seen:
#             seen.add(k)
#             out.append(it.strip())
#     return out

# def parse_authors(cell: Any) -> List[str]:
#     s = safe_str(cell)
#     if not s:
#         return []
#     try:
#         if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
#             v = json.loads(s)
#             if isinstance(v, list):
#                 return _dedupe([safe_str(x) for x in v if safe_str(x)])
#             if isinstance(v, dict) and isinstance(v.get("authors"), list):
#                 return _dedupe([safe_str(x) for x in v["authors"] if safe_str(x)])
#     except Exception:
#         pass
#     return _dedupe([p for p in SEP_RX.split(s) if p])

# # --------------------------- database clients --------------------------------

# SQLITE_DB = config.SQLITE_DB  # honor your D:\OSPO\KG-RAG1\researchers_fixed.db
# driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # --------------------------- read rows from sqlite ----------------------------

# def read_rows() -> List[Dict[str, Any]]:
#     conn = sqlite3.connect(SQLITE_DB)
#     cur = conn.cursor()

#     # Figure out available columns
#     cur.execute("PRAGMA table_info(research_info);")
#     cols = {r[1] for r in cur.fetchall()}
#     has_doi, has_pub = "doi" in cols, "publication_date" in cols

#     sql = "SELECT id, researcher_name, work_title, authors, info"
#     sql += ", doi" if has_doi else ", '' AS doi"
#     sql += ", publication_date" if has_pub else ", '' AS publication_date"
#     sql += " FROM research_info"
#     cur.execute(sql)
#     raw = cur.fetchall()
#     conn.close()

#     rows: List[Dict[str, Any]] = []
#     for pid, rname, title, authors, info, doi, pubdate in raw:
#         title = safe_str(title)
#         if not title:
#             continue
#         authors_list = parse_authors(authors)
#         primary = safe_str(rname) or (authors_list[0] if authors_list else "")
#         title_short = title[:2048]

#         rows.append({
#             "paper_id": str(pid),
#             "title": title,
#             "title_short": title_short,
#             "info": safe_str(info),
#             "primary_author": primary,
#             "co_authors": [a for a in authors_list if a and a.lower() != primary.lower()],
#             "year": extract_year(info, pubdate),
#             "doi_link": extract_doi_link(info, doi),
#         })
#     return rows

# # --------------------------- schema (CE friendly) -----------------------------

# def ensure_schema():
#     with driver.session(database=config.NEO4J_DB) as s:
#         # Uniqueness
#         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
#         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
#         s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")
#         # Helpful secondary indexes
#         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.year)")
#         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.doi_link)")
#         s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.title_short)")

# UPSERT = """
# UNWIND $rows AS row
# MERGE (p:Paper {paper_id: row.paper_id})
#   SET p.title       = row.title,
#       p.title_short = row.title_short,
#       p.info        = row.info,
#       p.year        = row.year,
#       p.doi_link    = row.doi_link

# WITH p, row
# WHERE row.primary_author IS NOT NULL AND row.primary_author <> ""
# MERGE (r:Researcher {name: row.primary_author})
# MERGE (a_primary:Author {name: row.primary_author})
# MERGE (r)-[:WROTE]->(p)
# MERGE (a_primary)-[:AUTHORED]->(p)
# MERGE (p)-[:HAS_RESEARCHER]->(r)
# MERGE (p)-[:HAS_AUTHOR]->(a_primary)

# WITH p, row
# UNWIND row.co_authors AS coName
# WITH p, trim(coName) AS cname
# WHERE cname <> ""
# MERGE (a:Author {name: cname})
# MERGE (a)-[:AUTHORED]->(p)
# MERGE (p)-[:HAS_AUTHOR]->(a)
# """

# COAUTHORS = """
# MATCH (p:Paper)<-[:HAS_AUTHOR]-(a1:Author),
#       (p)<-[:HAS_AUTHOR]-(a2:Author)
# WHERE id(a1) < id(a2)
# MERGE (a1)-[:COAUTHORED_WITH]-(a2)
# """

# # --------------------------- batched ingest ----------------------------------

# def ingest(rows: List[Dict[str, Any]], batch_size: int = 200):
#     batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

#     def worker(chunk):
#         with driver.session(database=config.NEO4J_DB) as s:
#             s.execute_write(lambda tx: tx.run(UPSERT, rows=chunk))
#         return len(chunk)

#     done, total = 0, len(rows)
#     with ThreadPoolExecutor(max_workers=config.PARALLEL_JOBS or 4) as ex:
#         futs = [ex.submit(worker, b) for b in batches]
#         for f in as_completed(futs):
#             done += f.result()
#             print(f"Committed {done}/{total} rows")

# def build_coauthor_edges():
#     with driver.session(database=config.NEO4J_DB) as s:
#         s.run(COAUTHORS)
#     print("Co-author edges built.")

# # --------------------------- main --------------------------------------------

# if __name__ == "__main__":
#     config.print_config_summary()
#     print(f"Using SQLite: {SQLITE_DB}")

#     try:
#         ensure_schema()
#     except ClientError as e:
#         print("Schema setup failed:", e)
#         raise

#     rows = read_rows()
#     print(f"Rows prepared: {len(rows)}")
#     if not rows:
#         print("No rows to ingest. Check your SQLite path and tables.")
#     else:
#         ingest(rows)
#         build_coauthor_edges()
#         print("✅ Done ingesting into Neo4j CE.")

# neo_ingest_full.py
"""
Neo4j Community Edition ingest (no vector indexes, no APOC required).

- Reads rows from SQLite (researchers_fixed.db) via config_full.SQLITE_DB
- Creates a compact paper/author/researcher graph
- Safe to re-run: uses MERGE and uniqueness constraints
- No hardcoded paths; all config comes from config_full.py

Nodes:
  (Paper {paper_id, title, title_short, info, year, doi_link})
  (Author {name})
  (Researcher {name})

Rels:
  (Author)-[:AUTHORED]->(Paper)
  (Paper)-[:HAS_AUTHOR]->(Author)
  (Researcher)-[:WROTE]->(Paper)
  (Paper)-[:HAS_RESEARCHER]->(Researcher)
  (Author)-[:COAUTHORED_WITH]-(Author)  # undirected (stored once)

Run:
  python neo_ingest_full.py
"""

import json
import math
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

import config_full as config

# --------------------------- regex helpers ------------------------------------

DATE_RE    = re.compile(r"\b(19|20)\d{2}\b")
PUBDATE_RE = re.compile(r"\b(?:Publication\s*Date|Date)\s*:\s*([0-9]{4})", re.I)
DOI_RE     = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
URL_RE     = re.compile(r"https?://\S+")
SEP_RX     = re.compile(r"\s*(?:;|,|\band\b|&|\|)\s*", re.I)

# --------------------------- neo4j connection ---------------------------------

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# --------------------------- small utils --------------------------------------

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        try:
            if math.isnan(x):
                return ""
        except Exception:
            return ""
    return str(x).strip()

def extract_year(info: str) -> Optional[int]:
    s = safe_str(info)
    if not s:
        return None
    m = PUBDATE_RE.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    m2 = DATE_RE.search(s)
    return int(m2.group(0)) if m2 else None

def extract_doi_link(info: str, doi: str) -> Optional[str]:
    doi = safe_str(doi)
    if doi:
        return f"https://doi.org/{doi}" if doi.lower().startswith("10.") else doi
    s = safe_str(info)
    if not s:
        return None
    m = DOI_RE.search(s)
    if m:
        return f"https://doi.org/{m.group(0)}"
    m2 = URL_RE.search(s)
    return m2.group(0) if m2 else None

def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        k = it.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out

def parse_authors(cell: Any) -> List[str]:
    s = safe_str(cell)
    if not s:
        return []
    try:
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            v = json.loads(s)
            if isinstance(v, list):
                return _dedupe([safe_str(x) for x in v if safe_str(x)])
            if isinstance(v, dict) and isinstance(v.get("authors"), list):
                return _dedupe([safe_str(x) for x in v["authors"] if safe_str(x)])
    except Exception:
        pass
    return _dedupe([p for p in SEP_RX.split(s) if p])

# --------------------------- read rows from sqlite ----------------------------

def read_rows() -> List[Dict[str, Any]]:
    db_path = config.SQLITE_DB
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(research_info);")
    cols = {r[1] for r in cur.fetchall()}
    has_doi, has_pub = "doi" in cols, "publication_date" in cols

    sql = "SELECT id, researcher_name, work_title, authors, info"
    sql += ", doi" if has_doi else ", '' AS doi"
    sql += ", publication_date" if has_pub else ", '' AS publication_date"
    sql += " FROM research_info"
    cur.execute(sql)
    raw = cur.fetchall()
    conn.close()

    rows: List[Dict[str, Any]] = []
    for pid, rname, title, authors, info, doi, pubdate in raw:
        title = safe_str(title)
        if not title:
            continue
        authors_list = parse_authors(authors)
        primary = safe_str(rname) or (authors_list[0] if authors_list else "")
        title_short = title[:2048]

        rows.append({
            "paper_id": str(pid),
            "title": title,
            "title_short": title_short,
            "info": safe_str(info),
            "primary_author": primary,
            "co_authors": [a for a in authors_list if a and a.lower() != primary.lower()],
            "year": extract_year(safe_str(pubdate) or safe_str(info)),
            "doi_link": extract_doi_link(info, doi),
        })
    return rows

# --------------------------- schema (CE-safe) ---------------------------------

def ensure_schema():
    with driver.session(database=config.NEO4J_DB) as s:
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")

# --------------------------- cypher upserts -----------------------------------

UPSERT = """
UNWIND $rows AS row
MERGE (p:Paper {paper_id: row.paper_id})
  SET p.title       = row.title,
      p.title_short = row.title_short,
      p.info        = row.info,
      p.year        = row.year,
      p.doi_link    = row.doi_link

WITH p, row
WHERE row.primary_author IS NOT NULL AND row.primary_author <> ""
MERGE (r:Researcher {name: row.primary_author})
MERGE (a_primary:Author {name: row.primary_author})
MERGE (r)-[:WROTE]->(p)
MERGE (a_primary)-[:AUTHORED]->(p)
MERGE (p)-[:HAS_RESEARCHER]->(r)
MERGE (p)-[:HAS_AUTHOR]->(a_primary)

WITH p, row
UNWIND row.co_authors AS coName
WITH p, trim(coName) AS cname
WHERE cname <> ""
MERGE (a:Author {name: cname})
MERGE (a)-[:AUTHORED]->(p)
MERGE (p)-[:HAS_AUTHOR]->(a)
"""

COAUTHORS = """
MATCH (p:Paper)<-[:HAS_AUTHOR]-(a1:Author),
      (p)<-[:HAS_AUTHOR]-(a2:Author)
WHERE id(a1) < id(a2)
MERGE (a1)-[:COAUTHORED_WITH]-(a2)
"""

# --------------------------- ingest runners -----------------------------------

def ingest(rows: List[Dict[str, Any]], batch_size: int = 200, workers: int = 8):
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    def worker(chunk):
        with driver.session(database=config.NEO4J_DB) as s:
            s.execute_write(lambda tx: tx.run(UPSERT, rows=chunk))
        return len(chunk)

    done, total = 0, len(rows)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, b) for b in batches]
        for f in as_completed(futures):
            done += f.result()
            print(f"Committed {done}/{total} rows")

def build_coauthor_edges():
    with driver.session(database=config.NEO4J_DB) as s:
        s.run(COAUTHORS)
    print("Co-author edges built.")

# --------------------------- main ---------------------------------------------

if __name__ == "__main__":
    print("— Neo4j CE Ingest —")
    print(f"SQLite : {config.SQLITE_DB}")
    print(f"Neo4j  : {config.NEO4J_URI} / db={config.NEO4J_DB}")

    try:
        ensure_schema()
    except ClientError as e:
        print("Schema setup failed:", e)
        raise

    rows = read_rows()
    print(f"Rows prepared: {len(rows)}")

    if not rows:
        print("No rows found. Check your SQLite database and table names.")
    else:
        ingest(rows, batch_size=int(getattr(config, "BATCH_SIZE", 200)), workers=int(getattr(config, "PARALLEL_JOBS", 4)))
        build_coauthor_edges()
        print("✅ Done ingesting into Neo4j Community Edition.")
