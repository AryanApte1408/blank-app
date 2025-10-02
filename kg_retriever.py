# # # kg_retriever.py
# # import re
# # from typing import List, Dict
# # from neo4j import GraphDatabase
# # import config_full as config

# # # --- Neo4j driver ---
# # driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # # --- Simple cleanup ---
# # def normalize_text(q: str) -> str:
# #     return re.sub(r"\s+", " ", q.strip())

# # # --- Query by author ---
# # def get_papers_by_author(author_name: str) -> List[Dict]:
# #     q = """
# #     MATCH (a:Author {name: $name})-[:AUTHORED]->(p:Paper)
# #     RETURN p.title AS title, p.year AS year, p.doi_link AS doi
# #     ORDER BY p.year DESC
# #     """
# #     with driver.session() as s:
# #         result = s.run(q, name=author_name)
# #         return [dict(r) for r in result]

# # # --- Query by researcher ---
# # def get_papers_by_researcher(rname: str) -> List[Dict]:
# #     q = """
# #     MATCH (r:Researcher {name: $name})-[:WROTE]->(p:Paper)
# #     RETURN p.title AS title, p.year AS year, p.doi_link AS doi
# #     ORDER BY p.year DESC
# #     """
# #     with driver.session() as s:
# #         result = s.run(q, name=rname)
# #         return [dict(r) for r in result]

# # # --- Query by paper title (fuzzy match using CONTAINS) ---
# # def get_paper_by_title(keyword: str) -> List[Dict]:
# #     q = """
# #     MATCH (p:Paper)
# #     WHERE toLower(p.title) CONTAINS toLower($kw)
# #     RETURN p.title AS title, p.year AS year, p.doi_link AS doi
# #     ORDER BY p.year DESC
# #     """
# #     with driver.session() as s:
# #         result = s.run(q, kw=keyword)
# #         return [dict(r) for r in result]

# # # --- Co-author graph for one author ---
# # def get_coauthors(author_name: str) -> List[str]:
# #     q = """
# #     MATCH (a:Author {name: $name})-[:COAUTHORED_WITH]-(co:Author)
# #     RETURN DISTINCT co.name AS name
# #     ORDER BY name
# #     """
# #     with driver.session() as s:
# #         result = s.run(q, name=author_name)
# #         return [r["name"] for r in result]

# # # --- Unified entrypoint ---
# # def query_kg(question: str) -> str:
# #     q = normalize_text(question)

# #     # Heuristics
# #     if q.lower().startswith("papers by "):
# #         author = q[10:].strip()
# #         res = get_papers_by_author(author)
# #         return f"Papers by {author}: {res}" if res else f"No papers found for {author}."

# #     if q.lower().startswith("researcher "):
# #         rname = q[11:].strip()
# #         res = get_papers_by_researcher(rname)
# #         return f"Papers by researcher {rname}: {res}" if res else f"No results for {rname}."

# #     if q.lower().startswith("find paper "):
# #         keyword = q[11:].strip()
# #         res = get_paper_by_title(keyword)
# #         return f"Papers matching '{keyword}': {res}" if res else f"No match for '{keyword}'."

# #     if q.lower().startswith("coauthors of "):
# #         author = q[12:].strip()
# #         res = get_coauthors(author)
# #         return f"Co-authors of {author}: {res}" if res else f"No co-authors found for {author}."

# #     return "⚠️ Query type not recognized. Try 'papers by <author>', 'researcher <name>', 'find paper <keyword>', or 'coauthors of <author>'."

# # # --- CLI test ---
# # if __name__ == "__main__":
# #     tests = [
# #         "papers by Sarah E. Woolf-King",
# #         "researcher Jeffrey Saltz",
# #         "find paper Risk Management",
# #         "coauthors of Ankita Juneja"
# #     ]
# #     for t in tests:
# #         print(f"\nQuery: {t}")
# #         print(query_kg(t))


# # kg_retriever.py
# """
# Lightweight Neo4j CE retriever (no APOC, no vector indexes).

# - Uses simple keyword matching (case-insensitive CONTAINS) over Paper titles,
#   Researcher names, and Author names.
# - Compatible with the graph created by neo_ingest_ce.py
# - No hardcoded paths: reads connection settings from config_full.py

# Public API:
#     query_graph(question: str, k: int = 10) -> List[dict]
#     get_papers_by_author(name: str, k: int = 25) -> List[dict]
#     get_papers_by_researcher(name: str, k: int = 25) -> List[dict]
#     get_paper_neighbors(title_or_id: str, k_authors: int = 10) -> dict
# """

# from typing import List, Dict, Tuple
# import re

# from neo4j import GraphDatabase
# import config_full as config

# driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # ------------------------------ helpers --------------------------------------

# _WS = re.compile(r"\s+")
# _ALNUM = re.compile(r"[A-Za-z0-9]+")

# def _norm(s: str) -> str:
#     return (_WS.sub(" ", s or "")).strip()

# def _keywords(q: str, min_len: int = 3, max_kw: int = 6) -> List[str]:
#     """Extract simple alphanumeric keywords from a query."""
#     toks = [t.lower() for t in _ALNUM.findall(q or "")]
#     # drop very short tokens and duplicates, keep order
#     seen, out = set(), []
#     for t in toks:
#         if len(t) >= min_len and t not in seen:
#             seen.add(t)
#             out.append(t)
#         if len(out) >= max_kw:
#             break
#     return out or ([_norm(q).lower()] if q else [])

# def _score_row(row: Dict, kws: List[str]) -> int:
#     """Very basic relevance scoring: count keyword hits across fields."""
#     hay = " ".join([
#         str(row.get("title") or ""),
#         " ".join(row.get("authors") or []),
#         str(row.get("researcher") or "")
#     ]).lower()
#     return sum(1 for kw in kws if kw in hay)

# def _run_query(cypher: str, **params) -> List[Dict]:
#     with driver.session(database=config.NEO4J_DB) as s:
#         return [dict(r) for r in s.run(cypher, **params)]

# # ------------------------------ core search ----------------------------------

# CY_SEARCH = """
# WITH $kws AS kws
# MATCH (p:Paper)
# OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)
# OPTIONAL MATCH (p)-[:HAS_RESEARCHER]->(r:Researcher)
# WITH p, collect(DISTINCT a.name) AS authors, r.name AS researcher, kws
# WHERE ANY(kw IN kws WHERE
#       toLower(p.title)       CONTAINS kw
#    OR toLower(coalesce(p.title_short,'')) CONTAINS kw
#    OR toLower(coalesce(p.info,''))        CONTAINS kw
#    OR toLower(coalesce(researcher,''))    CONTAINS kw
#    OR ANY(an IN authors WHERE toLower(coalesce(an,'')) CONTAINS kw)
# )
# RETURN p.paper_id AS paper_id,
#        p.title AS title,
#        p.year AS year,
#        p.doi_link AS doi_link,
#        authors AS authors,
#        researcher AS researcher
# LIMIT $hard_limit
# """

# def query_graph(question: str, k: int = 10) -> List[Dict]:
#     """
#     Keyword search over the KG.
#     Returns up to k best rows (local scored after a broader LIMIT).
#     """
#     kws = _keywords(question)
#     # Pull a wider pool (e.g., 5x) then score and slice locally
#     hard_limit = max(k * 5, 50)
#     rows = _run_query(CY_SEARCH, kws=kws, hard_limit=hard_limit)

#     # Local scoring (very simple) + stable ordering
#     for r in rows:
#         r["_score"] = _score_row(r, kws)
#     rows.sort(key=lambda x: (-x["_score"], x.get("year") if x.get("year") is not None else -9999, x.get("title","")))
#     return rows[:k]

# # ------------------------------ convenience APIs -----------------------------

# CY_BY_AUTHOR = """
# MATCH (a:Author {name: $name})-[:AUTHORED]->(p:Paper)
# OPTIONAL MATCH (p)-[:HAS_RESEARCHER]->(r:Researcher)
# WITH p, r
# OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(co:Author)
# WITH p, r, collect(DISTINCT co.name) AS authors
# RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year, p.doi_link AS doi_link,
#        authors AS authors, r.name AS researcher
# ORDER BY coalesce(p.year, -9999) DESC, p.title
# LIMIT $k
# """

# def get_papers_by_author(name: str, k: int = 25) -> List[Dict]:
#     return _run_query(CY_BY_AUTHOR, name=name, k=k)

# CY_BY_RESEARCHER = """
# MATCH (r:Researcher {name: $name})-[:WROTE]->(p:Paper)
# WITH p, r
# OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)
# RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year, p.doi_link AS doi_link,
#        collect(DISTINCT a.name) AS authors, r.name AS researcher
# ORDER BY coalesce(p.year, -9999) DESC, p.title
# LIMIT $k
# """

# def get_papers_by_researcher(name: str, k: int = 25) -> List[Dict]:
#     return _run_query(CY_BY_RESEARCHER, name=name, k=k)

# CY_NEIGHBORS = """
# // Accept either paper_id exact match or title (partial)
# WITH toLower($needle) AS needle
# MATCH (p:Paper)
# WHERE p.paper_id = $needle_raw OR toLower(p.title) CONTAINS needle
# WITH p
# OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)
# OPTIONAL MATCH (p)-[:HAS_RESEARCHER]->(r:Researcher)
# WITH p,
#      collect(DISTINCT a.name) AS authors,
#      r.name AS researcher
# OPTIONAL MATCH (a2:Author)-[:COAUTHORED_WITH]-(a)
# WITH p, authors, researcher, collect(DISTINCT a2.name) AS coauthors
# RETURN p.paper_id   AS paper_id,
#        p.title      AS title,
#        p.year       AS year,
#        p.doi_link   AS doi_link,
#        authors      AS authors,
#        researcher   AS researcher,
#        coauthors[0..$k_authors] AS coauthors
# LIMIT 1
# """

# def get_paper_neighbors(title_or_id: str, k_authors: int = 10) -> Dict:
#     rows = _run_query(CY_NEIGHBORS, needle=title_or_id.lower(), needle_raw=title_or_id, k_authors=k_authors)
#     return rows[0] if rows else {}

# # ------------------------------ script mode -----------------------------------

# if __name__ == "__main__":
#     print("— KG retriever sanity check —")
#     print(f"Neo4j: {config.NEO4J_URI} / db={config.NEO4J_DB}")
#     q = "theta oscillations memory hippocampus"
#     hits = query_graph(q, k=5)
#     for i, h in enumerate(hits, 1):
#         print(f"[{i}] {h.get('year')} | {h.get('title')} | {', '.join(h.get('authors') or [])}")


# kg_retriever.py
from typing import List, Dict
import re
from neo4j import GraphDatabase
import config_full as config

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# --- helpers ---
_WS = re.compile(r"\s+")
_ALNUM = re.compile(r"[A-Za-z0-9]+")

def _norm(s: str) -> str:
    return (_WS.sub(" ", s or "")).strip()

def _keywords(q: str, min_len: int = 3, max_kw: int = 6) -> List[str]:
    toks = [t.lower() for t in _ALNUM.findall(q or "")]
    seen, out = set(), []
    for t in toks:
        if len(t) >= min_len and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_kw:
            break
    return out or ([_norm(q).lower()] if q else [])

def _score_row(row: Dict, kws: List[str]) -> int:
    hay = " ".join([
        str(row.get("title") or ""),
        " ".join(row.get("authors") or []),
        str(row.get("researcher") or "")
    ]).lower()
    return sum(1 for kw in kws if kw in hay)

def _run_query(cypher: str, **params) -> List[Dict]:
    with driver.session(database=config.NEO4J_DB) as s:
        return [dict(r) for r in s.run(cypher, **params)]

# --- main search ---
CY_SEARCH = """
WITH $kws AS kws
MATCH (p:Paper)
OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)
OPTIONAL MATCH (p)-[:HAS_RESEARCHER]->(r:Researcher)
WITH p, collect(DISTINCT a.name) AS authors, r.name AS researcher, kws
WHERE ANY(kw IN kws WHERE
      toLower(p.title) CONTAINS kw
   OR ANY(an IN authors WHERE toLower(coalesce(an,'')) CONTAINS kw)
   OR toLower(coalesce(researcher,'')) CONTAINS kw)
RETURN p.paper_id AS paper_id,
       p.title AS title,
       p.year AS year,
       p.doi_link AS doi_link,
       authors AS authors,
       researcher AS researcher
LIMIT $hard_limit
"""

def query_graph(question: str, k: int = 10) -> List[Dict]:
    kws = _keywords(question)
    rows = _run_query(CY_SEARCH, kws=kws, hard_limit=max(k*5, 50))
    for r in rows:
        r["_score"] = _score_row(r, kws)
    rows.sort(key=lambda x: (-x["_score"], x.get("year") or -9999, x.get("title","")))
    return rows[:k]

# --- convenience APIs ---
CY_BY_AUTHOR = """
MATCH (a:Author {name: $name})-[:AUTHORED]->(p:Paper)
OPTIONAL MATCH (p)-[:HAS_RESEARCHER]->(r:Researcher)
RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year,
       p.doi_link AS doi_link, collect(DISTINCT a.name) AS authors, r.name AS researcher
ORDER BY coalesce(p.year, -9999) DESC
LIMIT $k
"""

def get_papers_by_author(name: str, k: int = 25) -> List[Dict]:
    return _run_query(CY_BY_AUTHOR, name=name, k=k)

CY_BY_RESEARCHER = """
MATCH (r:Researcher {name: $name})-[:WROTE]->(p:Paper)
OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)
RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year,
       p.doi_link AS doi_link, collect(DISTINCT a.name) AS authors, r.name AS researcher
ORDER BY coalesce(p.year, -9999) DESC
LIMIT $k
"""

def get_papers_by_researcher(name: str, k: int = 25) -> List[Dict]:
    return _run_query(CY_BY_RESEARCHER, name=name, k=k)
