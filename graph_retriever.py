# # # # # # # # from kg_retriever import query_graph, get_papers_by_author, get_papers_by_researcher

# # # # # # # # def search_graph(query: str, k: int = 10):
# # # # # # # #     return query_graph(query, k=k)

# # # # # # # # def papers_by_author(name: str, k: int = 25):
# # # # # # # #     return get_papers_by_author(name, k=k)

# # # # # # # # def papers_by_researcher(name: str, k: int = 25):
# # # # # # # #     return get_papers_by_researcher(name, k=k)


# # # # # # # # graph_retriever.py
# # # # # # # from neo4j import GraphDatabase
# # # # # # # import config_full as config
# # # # # # # import re

# # # # # # # # ------------------ Neo4j setup ------------------
# # # # # # # driver = GraphDatabase.driver(
# # # # # # #     config.NEO4J_URI,
# # # # # # #     auth=(config.NEO4J_USER, config.NEO4J_PASS)
# # # # # # # )
# # # # # # # DB_NAME = getattr(config, "NEO4J_DB", "neo4j")

# # # # # # # # ------------------ helpers ------------------
# # # # # # # def _safe_str(v):
# # # # # # #     return str(v).strip() if v else ""

# # # # # # # def _normalize(txt: str):
# # # # # # #     return re.sub(r"[^a-z0-9 ]+", "", _safe_str(txt).lower())

# # # # # # # def _similarity(a, b):
# # # # # # #     a, b = set(_normalize(a).split()), set(_normalize(b).split())
# # # # # # #     if not a or not b:
# # # # # # #         return 0
# # # # # # #     return len(a & b) / len(a | b)

# # # # # # # # ------------------ main graph search ------------------
# # # # # # # def search_graph(query_text: str, k: int = 5):
# # # # # # #     """
# # # # # # #     Retrieve related graph nodes based on fuzzy match against title, researcher, or authors.
# # # # # # #     Expands context with RELATED_TO / CITED_BY links.
# # # # # # #     """
# # # # # # #     q_norm = _normalize(query_text)
# # # # # # #     if not q_norm:
# # # # # # #         return []

# # # # # # #     cypher = """
# # # # # # #     MATCH (p:Paper)
# # # # # # #     OPTIONAL MATCH (p)-[:AUTHORED_BY]->(r:Researcher)
# # # # # # #     OPTIONAL MATCH (p)-[:CITED_BY|:RELATED_TO]->(related)
# # # # # # #     RETURN p.title AS title,
# # # # # # #            p.year AS year,
# # # # # # #            p.doi AS doi,
# # # # # # #            p.info AS info,
# # # # # # #            r.name AS researcher,
# # # # # # #            COLLECT(DISTINCT r.name) AS authors,
# # # # # # #            COLLECT(DISTINCT related.title) AS related_papers
# # # # # # #     """

# # # # # # #     results = []
# # # # # # #     try:
# # # # # # #         with driver.session(database=DB_NAME) as session:
# # # # # # #             data = session.run(cypher)
# # # # # # #             for row in data:
# # # # # # #                 title = _safe_str(row["title"])
# # # # # # #                 researcher = _safe_str(row["researcher"])
# # # # # # #                 authors = row["authors"] or []
# # # # # # #                 info = _safe_str(row["info"])
# # # # # # #                 doi = _safe_str(row["doi"])
# # # # # # #                 year = _safe_str(row["year"])
# # # # # # #                 related = row["related_papers"] or []

# # # # # # #                 score = max(
# # # # # # #                     _similarity(q_norm, title),
# # # # # # #                     _similarity(q_norm, researcher),
# # # # # # #                     *[_similarity(q_norm, a) for a in authors]
# # # # # # #                 )

# # # # # # #                 if score > 0.1:
# # # # # # #                     results.append({
# # # # # # #                         "title": title,
# # # # # # #                         "researcher": researcher,
# # # # # # #                         "authors": authors,
# # # # # # #                         "year": year,
# # # # # # #                         "doi": doi,
# # # # # # #                         "info": info,
# # # # # # #                         "related": related,
# # # # # # #                         "score": score
# # # # # # #                     })
# # # # # # #         results.sort(key=lambda x: x["score"], reverse=True)
# # # # # # #         return results[:k]

# # # # # # #     except Exception as e:
# # # # # # #         print(f"❌ Neo4j query error: {e}")
# # # # # # #         return []

# # # # # # # if __name__ == "__main__":
# # # # # # #     q = "Jeffrey Saltz"
# # # # # # #     hits = search_graph(q, k=10)
# # # # # # #     for h in hits:
# # # # # # #         print(f"{h['score']:.2f} | {h['title']} — {h['researcher']}")

# # # # # # # graph_retriever.py
# # # # # # from neo4j import GraphDatabase
# # # # # # import config_full as config
# # # # # # import re

# # # # # # driver = GraphDatabase.driver(
# # # # # #     config.NEO4J_URI,
# # # # # #     auth=(config.NEO4J_USER, config.NEO4J_PASS)
# # # # # # )
# # # # # # DB_NAME = getattr(config, "NEO4J_DB", "neo4j")

# # # # # # # ------------------ helpers ------------------
# # # # # # def _safe_str(v): return str(v).strip() if v else ""
# # # # # # def _normalize(txt: str): return re.sub(r"[^a-z0-9 ]+", "", _safe_str(txt).lower())
# # # # # # def _similarity(a, b):
# # # # # #     a, b = set(_normalize(a).split()), set(_normalize(b).split())
# # # # # #     if not a or not b: return 0
# # # # # #     return len(a & b) / len(a | b)

# # # # # # # ------------------ graph retrieval ------------------
# # # # # # def expand_graph_context(researcher=None, title=None, doi=None, hops: int = 1):
# # # # # #     """
# # # # # #     Pull a small subgraph around the node(s) identified by researcher/title/doi.
# # # # # #     Fixes:
# # # # # #       - Use p.doi_link (your ingest writes this), not p.doi
# # # # # #       - Drop unknown rels (RELATED_TO, CITED_BY)
# # # # # #       - Parameterized WHERE to avoid injection
# # # # # #       - Accept either DOI URL or suffix in 'doi'
# # # # # #     """
# # # # # #     params = {}
# # # # # #     where = []
# # # # # #     match = ["MATCH (p:Paper)"]

# # # # # #     if researcher:
# # # # # #         # Prefer WROTE; HAS_RESEARCHER also exists in your ingest
# # # # # #         match = ["MATCH (r:Researcher {name: $researcher})-[:WROTE|HAS_RESEARCHER]->(p:Paper)"]
# # # # # #         params["researcher"] = researcher
# # # # # #     elif title:
# # # # # #         where.append("toLower(p.title) CONTAINS toLower($title)")
# # # # # #         params["title"] = title
# # # # # #     elif doi:
# # # # # #         # If doi starts with http, compare full; else match by doi suffix
# # # # # #         where.append("""
# # # # # #             CASE
# # # # # #               WHEN $doi STARTS WITH 'http' THEN toLower(p.doi_link) = toLower($doi)
# # # # # #               ELSE toLower(p.doi_link) ENDS WITH toLower($doi)
# # # # # #             END
# # # # # #         """)
# # # # # #         params["doi"] = doi
# # # # # #     else:
# # # # # #         return []

# # # # # #     rels = "OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)"

# # # # # #     cypher = f"""
# # # # # #     {' '.join(match)}
# # # # # #     {'WHERE ' + ' AND '.join(where) if where else ''}
# # # # # #     {rels}
# # # # # #     RETURN DISTINCT
# # # # # #         p.title AS title,
# # # # # #         p.year  AS year,
# # # # # #         p.doi_link AS doi,
# # # # # #         collect(DISTINCT a.name) AS authors,
# # # # # #         [] AS related,
# # # # # #         p.info AS info,
# # # # # #         $researcher AS researcher
# # # # # #     LIMIT 25
# # # # # #     """

# # # # # #     # Ensure the parameter exists for RETURN even if not filtering by researcher
# # # # # #     if "researcher" not in params:
# # # # # #         params["researcher"] = researcher or ""

# # # # # #     try:
# # # # # #         with driver.session(database=DB_NAME) as s:
# # # # # #             rows = [dict(r) for r in s.run(cypher, **params)]
# # # # # #         return rows
# # # # # #     except Exception as e:
# # # # # #         print("Graph expansion error:", e)
# # # # # #         return []

# # # # # # def weighted_search(query_text: str, candidates, w_r=0.6, w_t=0.25, w_a=0.1, w_rel=0.05):
# # # # # #     q_norm = _normalize(query_text)
# # # # # #     out = []
# # # # # #     for row in candidates:
# # # # # #         score = (
# # # # # #             w_r * _similarity(q_norm, row.get("researcher", "")) +
# # # # # #             w_t * _similarity(q_norm, row.get("title", "")) +
# # # # # #             w_a * sum(_similarity(q_norm, a) for a in row.get("authors", [])) / (len(row.get("authors", [])) or 1) +
# # # # # #             w_rel * sum(_similarity(q_norm, r) for r in row.get("related", [])) / (len(row.get("related", [])) or 1)
# # # # # #         )
# # # # # #         row["score"] = round(score, 3)
# # # # # #         out.append(row)
# # # # # #     out.sort(key=lambda x: x["score"], reverse=True)
# # # # # #     return out

# # # # # # def search_graph_from_chroma_meta(query_text, chroma_metas, k=8):
# # # # # #     """
# # # # # #     Restrict Neo4j exploration to nodes surfaced by Chroma metadata.
# # # # # #     """
# # # # # #     all_rows = []
# # # # # #     for meta in chroma_metas:
# # # # # #         rname = _safe_str(meta.get("researcher"))
# # # # # #         title = _safe_str(meta.get("title"))
# # # # # #         doi = _safe_str(meta.get("doi"))
# # # # # #         rows = expand_graph_context(rname, title, doi)
# # # # # #         all_rows.extend(rows)
# # # # # #     ranked = weighted_search(query_text, all_rows)
# # # # # #     return ranked[:k]


# # # # # # graph_retriever.py
# # # # # from typing import List, Dict
# # # # # import re
# # # # # from neo4j import GraphDatabase
# # # # # import config_full as config

# # # # # # ------------------ connection ------------------
# # # # # driver = GraphDatabase.driver(
# # # # #     config.NEO4J_URI,
# # # # #     auth=(config.NEO4J_USER, config.NEO4J_PASS)
# # # # # )
# # # # # DB_NAME = getattr(config, "NEO4J_DB", "neo4j")

# # # # # # ------------------ helpers ------------------
# # # # # _WS = re.compile(r"\s+")
# # # # # _ALNUM = re.compile(r"[A-Za-z0-9]+")

# # # # # def _safe_str(v):
# # # # #     return str(v).strip() if v else ""

# # # # # def _normalize(txt: str) -> str:
# # # # #     return re.sub(r"[^a-z0-9 ]+", "", _safe_str(txt).lower())

# # # # # def _similarity(a, b):
# # # # #     a, b = set(_normalize(a).split()), set(_normalize(b).split())
# # # # #     if not a or not b:
# # # # #         return 0
# # # # #     return len(a & b) / len(a | b)

# # # # # def _norm(s: str) -> str:
# # # # #     return (_WS.sub(" ", s or "")).strip()

# # # # # def _keywords(q: str, min_len: int = 3, max_kw: int = 6) -> List[str]:
# # # # #     toks = [t.lower() for t in _ALNUM.findall(q or "")]
# # # # #     seen, out = set(), []
# # # # #     for t in toks:
# # # # #         if len(t) >= min_len and t not in seen:
# # # # #             seen.add(t)
# # # # #             out.append(t)
# # # # #         if len(out) >= max_kw:
# # # # #             break
# # # # #     return out or ([_norm(q).lower()] if q else [])

# # # # # def _score_row(row: Dict, kws: List[str]) -> int:
# # # # #     hay = " ".join([
# # # # #         str(row.get("title") or ""),
# # # # #         " ".join(row.get("authors") or []),
# # # # #         str(row.get("researcher") or "")
# # # # #     ]).lower()
# # # # #     return sum(1 for kw in kws if kw in hay)

# # # # # def _run_query(cypher: str, **params) -> List[Dict]:
# # # # #     with driver.session(database=DB_NAME) as s:
# # # # #         return [dict(r) for r in s.run(cypher, **params)]

# # # # # # ------------------ graph retrieval ------------------
# # # # # def expand_graph_context(researcher=None, title=None, doi=None, hops: int = 1):
# # # # #     """
# # # # #     Pull a small subgraph around the node(s) identified by researcher/title/doi.

# # # # #     Fixes:
# # # # #       - Use p.doi_link (ingest writes this), not p.doi
# # # # #       - Drop unknown rels (RELATED_TO, CITED_BY)
# # # # #       - Parameterized WHERE to avoid injection
# # # # #       - Accept either DOI URL or suffix in 'doi'
# # # # #       - Return paper_id so the UI can render exactly what was retrieved
# # # # #     """
# # # # #     params = {}
# # # # #     where = []
# # # # #     match = ["MATCH (p:Paper)"]

# # # # #     if researcher:
# # # # #         match = ["MATCH (r:Researcher {name: $researcher})-[:WROTE|HAS_RESEARCHER]->(p:Paper)"]
# # # # #         params["researcher"] = researcher
# # # # #     elif title:
# # # # #         where.append("toLower(p.title) CONTAINS toLower($title)")
# # # # #         params["title"] = title
# # # # #     elif doi:
# # # # #         where.append("""
# # # # #             CASE
# # # # #               WHEN $doi STARTS WITH 'http' THEN toLower(p.doi_link) = toLower($doi)
# # # # #               ELSE toLower(p.doi_link) ENDS WITH toLower($doi)
# # # # #             END
# # # # #         """)
# # # # #         params["doi"] = doi
# # # # #     else:
# # # # #         return []

# # # # #     rels = "OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)"

# # # # #     cypher = f"""
# # # # #     {' '.join(match)}
# # # # #     {'WHERE ' + ' AND '.join(where) if where else ''}
# # # # #     {rels}
# # # # #     RETURN DISTINCT
# # # # #         p.paper_id AS paper_id,
# # # # #         p.title    AS title,
# # # # #         p.year     AS year,
# # # # #         p.doi_link AS doi,
# # # # #         collect(DISTINCT a.name) AS authors,
# # # # #         [] AS related,
# # # # #         p.info AS info,
# # # # #         $researcher AS researcher
# # # # #     LIMIT 25
# # # # #     """

# # # # #     if "researcher" not in params:
# # # # #         params["researcher"] = researcher or ""

# # # # #     try:
# # # # #         with driver.session(database=DB_NAME) as s:
# # # # #             rows = [dict(r) for r in s.run(cypher, **params)]
# # # # #         return rows
# # # # #     except Exception as e:
# # # # #         print("Graph expansion error:", e)
# # # # #         return []

# # # # # def weighted_search(query_text: str, candidates, w_r=0.6, w_t=0.25, w_a=0.1, w_rel=0.05):
# # # # #     q_norm = _normalize(query_text)
# # # # #     out = []
# # # # #     for row in candidates:
# # # # #         score = (
# # # # #             w_r * _similarity(q_norm, row.get("researcher", "")) +
# # # # #             w_t * _similarity(q_norm, row.get("title", "")) +
# # # # #             w_a * sum(_similarity(q_norm, a) for a in row.get("authors", [])) / (len(row.get("authors", [])) or 1) +
# # # # #             w_rel * sum(_similarity(q_norm, r) for r in row.get("related", [])) / (len(row.get("related", [])) or 1)
# # # # #         )
# # # # #         row["score"] = round(score, 3)
# # # # #         out.append(row)
# # # # #     out.sort(key=lambda x: x["score"], reverse=True)
# # # # #     return out

# # # # # def search_graph_from_chroma_meta(query_text, chroma_metas, k=8):
# # # # #     """
# # # # #     Restrict Neo4j exploration to nodes surfaced by Chroma metadata.
# # # # #     """
# # # # #     all_rows = []
# # # # #     for meta in chroma_metas:
# # # # #         rname = _safe_str(meta.get("researcher"))
# # # # #         title = _safe_str(meta.get("title"))
# # # # #         doi = _safe_str(meta.get("doi"))
# # # # #         rows = expand_graph_context(rname, title, doi)
# # # # #         all_rows.extend(rows)
# # # # #     ranked = weighted_search(query_text, all_rows)
# # # # #     return ranked[:k]

# # # # # # ------------------ additional queries (keyword-based) ------------------
# # # # # CY_SEARCH = """
# # # # # WITH $kws AS kws
# # # # # MATCH (p:Paper)
# # # # # OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)
# # # # # OPTIONAL MATCH (p)-[:HAS_RESEARCHER]->(r:Researcher)
# # # # # WITH p, collect(DISTINCT a.name) AS authors, r.name AS researcher, kws
# # # # # WHERE ANY(kw IN kws WHERE
# # # # #       toLower(p.title) CONTAINS kw
# # # # #    OR ANY(an IN authors WHERE toLower(coalesce(an,'')) CONTAINS kw)
# # # # #    OR toLower(coalesce(researcher,'')) CONTAINS kw)
# # # # # RETURN p.paper_id AS paper_id,
# # # # #        p.title AS title,
# # # # #        p.year AS year,
# # # # #        p.doi_link AS doi_link,
# # # # #        authors AS authors,
# # # # #        researcher AS researcher
# # # # # LIMIT $hard_limit
# # # # # """

# # # # # def query_graph(question: str, k: int = 10) -> List[Dict]:
# # # # #     kws = _keywords(question)
# # # # #     rows = _run_query(CY_SEARCH, kws=kws, hard_limit=max(k*5, 50))
# # # # #     for r in rows:
# # # # #         r["_score"] = _score_row(r, kws)
# # # # #     rows.sort(key=lambda x: (-x["_score"], x.get("year") or -9999, x.get("title","")))
# # # # #     return rows[:k]

# # # # # CY_BY_AUTHOR = """
# # # # # MATCH (a:Author {name: $name})-[:AUTHORED]->(p:Paper)
# # # # # OPTIONAL MATCH (p)-[:HAS_RESEARCHER]->(r:Researcher)
# # # # # RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year,
# # # # #        p.doi_link AS doi_link, collect(DISTINCT a.name) AS authors, r.name AS researcher
# # # # # ORDER BY coalesce(p.year, -9999) DESC
# # # # # LIMIT $k
# # # # # """

# # # # # def get_papers_by_author(name: str, k: int = 25) -> List[Dict]:
# # # # #     return _run_query(CY_BY_AUTHOR, name=name, k=k)

# # # # # CY_BY_RESEARCHER = """
# # # # # MATCH (r:Researcher {name: $name})-[:WROTE]->(p:Paper)
# # # # # OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)
# # # # # RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year,
# # # # #        p.doi_link AS doi_link, collect(DISTINCT a.name) AS authors, r.name AS researcher
# # # # # ORDER BY coalesce(p.year, -9999) DESC
# # # # # LIMIT $k
# # # # # """

# # # # # def get_papers_by_researcher(name: str, k: int = 25) -> List[Dict]:
# # # # #     return _run_query(CY_BY_RESEARCHER, name=name, k=k)

# # # # """
# # # # graph_retriever.py - Database-agnostic graph retrieval
# # # # Queries DatabaseManager for connection info
# # # # """
# # # # from typing import List, Dict, Optional
# # # # import re
# # # # from neo4j import GraphDatabase
# # # # from database_manager import get_active_db_config

# # # # _driver = None


# # # # def get_neo4j_driver():
# # # #     """Get Neo4j driver from active database config."""
# # # #     global _driver
# # # #     config = get_active_db_config()
    
# # # #     # Reconnect if config changed
# # # #     if _driver is None:
# # # #         _driver = GraphDatabase.driver(
# # # #             config.neo4j_uri,
# # # #             auth=(config.neo4j_user, config.neo4j_password)
# # # #         )
    
# # # #     return _driver


# # # # def close_driver():
# # # #     """Close Neo4j driver (for cleanup or reconnection)."""
# # # #     global _driver
# # # #     if _driver:
# # # #         _driver.close()
# # # #         _driver = None


# # # # _WS = re.compile(r"\s+")
# # # # _ALNUM = re.compile(r"[A-Za-z0-9]+")


# # # # def _safe_str(v):
# # # #     return str(v).strip() if v else ""


# # # # def _safe_int(v):
# # # #     try:
# # # #         return int(v) if v else 0
# # # #     except (ValueError, TypeError):
# # # #         return 0


# # # # def _normalize(txt: str) -> str:
# # # #     return re.sub(r"[^a-z0-9 ]+", "", _safe_str(txt).lower())


# # # # def _similarity(a, b):
# # # #     a, b = set(_normalize(a).split()), set(_normalize(b).split())
# # # #     if not a or not b:
# # # #         return 0
# # # #     return len(a & b) / len(a | b)


# # # # def _norm(s: str) -> str:
# # # #     return (_WS.sub(" ", s or "")).strip()


# # # # def _keywords(q: str, min_len: int = 3, max_kw: int = 6) -> List[str]:
# # # #     toks = [t.lower() for t in _ALNUM.findall(q or "")]
# # # #     seen, out = set(), []
# # # #     for t in toks:
# # # #         if len(t) >= min_len and t not in seen:
# # # #             seen.add(t)
# # # #             out.append(t)
# # # #         if len(out) >= max_kw:
# # # #             break
# # # #     return out or ([_norm(q).lower()] if q else [])


# # # # def _score_row(row: Dict, kws: List[str]) -> int:
# # # #     hay = " ".join([
# # # #         str(row.get("title") or ""),
# # # #         " ".join(row.get("authors") or []),
# # # #         str(row.get("researcher") or "")
# # # #     ]).lower()
# # # #     return sum(1 for kw in kws if kw in hay)


# # # # def _run_query(cypher: str, **params) -> List[Dict]:
# # # #     """Execute Cypher query using active database config."""
# # # #     try:
# # # #         config = get_active_db_config()
# # # #         driver = get_neo4j_driver()
        
# # # #         with driver.session(database=config.neo4j_database) as s:
# # # #             return [dict(r) for r in s.run(cypher, **params)]
# # # #     except Exception as e:
# # # #         print(f"❌ Neo4j query error: {e}")
# # # #         return []


# # # # def expand_graph_context(
# # # #     researcher: Optional[str] = None,
# # # #     title: Optional[str] = None,
# # # #     doi: Optional[str] = None,
# # # #     hops: int = 1
# # # # ) -> List[Dict]:
# # # #     """
# # # #     Pull subgraph - schema detection based on active database mode.
# # # #     Automatically adapts to full vs abstracts schema.
# # # #     """
# # # #     config = get_active_db_config()
# # # #     mode = config.mode
    
# # # #     params = {}
# # # #     where = []
# # # #     match = ["MATCH (p:Paper)"]

# # # #     if mode == "abstracts":
# # # #         # Abstracts schema: Researcher-AUTHORED->Paper
# # # #         if researcher:
# # # #             match = ["MATCH (r:Researcher {name: $researcher})-[:AUTHORED]->(p:Paper)"]
# # # #             params["researcher"] = researcher
# # # #         elif title:
# # # #             where.append("toLower(p.title) CONTAINS toLower($title)")
# # # #             params["title"] = title
# # # #         elif doi:
# # # #             where.append("(toLower(p.doi) = toLower($doi) OR toLower(p.doi) ENDS WITH toLower($doi))")
# # # #             params["doi"] = doi
# # # #         else:
# # # #             return []

# # # #         rels = """
# # # #         OPTIONAL MATCH (s:Source)-[:PUBLISHED]->(p)
# # # #         OPTIONAL MATCH (r:Researcher)-[:AUTHORED]->(p)
# # # #         """

# # # #         cypher = f"""
# # # #         {' '.join(match)}
# # # #         {'WHERE ' + ' AND '.join(where) if where else ''}
# # # #         {rels}
# # # #         RETURN DISTINCT
# # # #             coalesce(p.doi, '') AS paper_id,
# # # #             coalesce(p.title, 'Untitled') AS title,
# # # #             coalesce(p.year, 0) AS year,
# # # #             coalesce(p.doi, '') AS doi,
# # # #             [] AS authors,
# # # #             [] AS related,
# # # #             coalesce(p.info, '') AS info,
# # # #             coalesce(r.name, s.name, 'Unknown') AS researcher,
# # # #             coalesce(s.name, '') AS source
# # # #         ORDER BY coalesce(p.year, 0) DESC
# # # #         LIMIT 25
# # # #         """
        
# # # #         if "researcher" not in params:
# # # #             params["researcher"] = researcher or ""
            
# # # #     else:
# # # #         # Full schema: Researcher-WROTE->Paper-HAS_AUTHOR->Author
# # # #         if researcher:
# # # #             match = ["MATCH (r:Researcher {name: $researcher})-[:WROTE]->(p:Paper)"]
# # # #             params["researcher"] = researcher
# # # #         elif title:
# # # #             where.append("toLower(p.title) CONTAINS toLower($title)")
# # # #             params["title"] = title
# # # #         elif doi:
# # # #             where.append("(toLower(p.doi_link) = toLower($doi) OR toLower(p.doi_link) ENDS WITH toLower($doi))")
# # # #             params["doi"] = doi
# # # #         else:
# # # #             return []

# # # #         rels = """
# # # #         OPTIONAL MATCH (p)-[:HAS_AUTHOR]->(a:Author)
# # # #         WITH p, collect(DISTINCT a.name)[..20] AS authors
# # # #         """

# # # #         cypher = f"""
# # # #         {' '.join(match)}
# # # #         {'WHERE ' + ' AND '.join(where) if where else ''}
# # # #         {rels}
# # # #         RETURN DISTINCT
# # # #             coalesce(p.paper_id, '') AS paper_id,
# # # #             coalesce(p.title, 'Untitled') AS title,
# # # #             coalesce(p.year, 0) AS year,
# # # #             coalesce(p.doi_link, '') AS doi,
# # # #             authors,
# # # #             [] AS related,
# # # #             coalesce(p.info, '') AS info,
# # # #             coalesce($researcher, '') AS researcher,
# # # #             '' AS source
# # # #         ORDER BY coalesce(p.year, 0) DESC
# # # #         LIMIT 25
# # # #         """
        
# # # #         if "researcher" not in params:
# # # #             params["researcher"] = researcher or ""

# # # #     return _run_query(cypher, **params)


# # # # def weighted_search(query_text: str, candidates, w_r=0.6, w_t=0.25, w_a=0.1, w_rel=0.05):
# # # #     """Rank papers with recency bias."""
# # # #     q_norm = _normalize(query_text)
# # # #     out = []
# # # #     current_year = 2025
    
# # # #     for row in candidates:
# # # #         score = (
# # # #             w_r * _similarity(q_norm, row.get("researcher", "")) +
# # # #             w_t * _similarity(q_norm, row.get("title", "")) +
# # # #             w_a * sum(_similarity(q_norm, a) for a in row.get("authors", [])) / max(len(row.get("authors", [])), 1) +
# # # #             w_rel * sum(_similarity(q_norm, r) for r in row.get("related", [])) / max(len(row.get("related", [])), 1)
# # # #         )
        
# # # #         year = _safe_int(row.get("year", 0))
# # # #         if year > 0:
# # # #             recency_bonus = max(0, (year - (current_year - 5)) / 5 * 0.2)
# # # #             score += recency_bonus
        
# # # #         row["score"] = round(score, 3)
# # # #         row["year"] = year
# # # #         out.append(row)
    
# # # #     out.sort(key=lambda x: (-x["score"], -x.get("year", 0)))
# # # #     return out


# # # # def search_graph_from_chroma_meta(query_text, chroma_metas, k=8):
# # # #     """Restrict Neo4j exploration to Chroma-identified papers."""
# # # #     all_rows = []
# # # #     seen_papers = set()
    
# # # #     for meta in chroma_metas:
# # # #         rname = _safe_str(meta.get("researcher"))
# # # #         title = _safe_str(meta.get("title"))
# # # #         doi = _safe_str(meta.get("doi"))
        
# # # #         rows = expand_graph_context(rname, title, doi)
        
# # # #         for row in rows:
# # # #             pid = row.get("paper_id", "")
# # # #             if pid and pid not in seen_papers:
# # # #                 seen_papers.add(pid)
# # # #                 all_rows.append(row)
    
# # # #     ranked = weighted_search(query_text, all_rows)
# # # #     return ranked[:k]


# # # # def query_graph(question: str, k: int = 10) -> List[Dict]:
# # # #     """Search graph with auto-detected schema."""
# # # #     config = get_active_db_config()
# # # #     mode = config.mode
    
# # # #     kws = _keywords(question)
    
# # # #     if mode == "abstracts":
# # # #         cypher = """
# # # #         WITH $kws AS kws
# # # #         MATCH (p:Paper)
# # # #         OPTIONAL MATCH (r:Researcher)-[:AUTHORED]->(p)
# # # #         OPTIONAL MATCH (s:Source)-[:PUBLISHED]->(p)
# # # #         WITH p, r.name AS researcher, s.name AS source, kws
# # # #         WHERE ANY(kw IN kws WHERE
# # # #               toLower(p.title) CONTAINS kw
# # # #            OR toLower(coalesce(p.abstract,'')) CONTAINS kw
# # # #            OR toLower(coalesce(researcher,'')) CONTAINS kw
# # # #            OR toLower(coalesce(source,'')) CONTAINS kw)
# # # #         RETURN coalesce(p.doi, '') AS paper_id,
# # # #                coalesce(p.title, 'Untitled') AS title,
# # # #                coalesce(p.year, 0) AS year,
# # # #                coalesce(p.doi, '') AS doi_link,
# # # #                [] AS authors,
# # # #                coalesce(researcher, source, 'Unknown') AS researcher
# # # #         ORDER BY coalesce(p.year, 0) DESC
# # # #         LIMIT $hard_limit
# # # #         """
# # # #     else:
# # # #         cypher = """
# # # #         WITH $kws AS kws
# # # #         MATCH (p:Paper)
# # # #         OPTIONAL MATCH (p)-[:HAS_AUTHOR]->(a:Author)
# # # #         OPTIONAL MATCH (p)-[:HAS_RESEARCHER]->(r:Researcher)
# # # #         WITH p, collect(DISTINCT a.name)[..20] AS authors, r.name AS researcher, kws
# # # #         WHERE ANY(kw IN kws WHERE
# # # #               toLower(p.title) CONTAINS kw
# # # #            OR ANY(an IN authors WHERE toLower(coalesce(an,'')) CONTAINS kw)
# # # #            OR toLower(coalesce(researcher,'')) CONTAINS kw)
# # # #         RETURN coalesce(p.paper_id, '') AS paper_id,
# # # #                coalesce(p.title, 'Untitled') AS title,
# # # #                coalesce(p.year, 0) AS year,
# # # #                coalesce(p.doi_link, '') AS doi_link,
# # # #                authors,
# # # #                coalesce(researcher, '') AS researcher
# # # #         ORDER BY coalesce(p.year, 0) DESC
# # # #         LIMIT $hard_limit
# # # #         """
    
# # # #     rows = _run_query(cypher, kws=kws, hard_limit=max(k*5, 50))
    
# # # #     for r in rows:
# # # #         r["_score"] = _score_row(r, kws)
    
# # # #     rows.sort(key=lambda x: (-x["_score"], -x.get("year", 0)))
# # # #     return rows[:k]


# # # # def get_papers_by_researcher(name: str, k: int = 25) -> List[Dict]:
# # # #     """Get papers by researcher (auto-detects schema)."""
# # # #     config = get_active_db_config()
# # # #     mode = config.mode
    
# # # #     if mode == "abstracts":
# # # #         cypher = """
# # # #         MATCH (r:Researcher {name: $name})-[:AUTHORED]->(p:Paper)
# # # #         OPTIONAL MATCH (s:Source)-[:PUBLISHED]->(p)
# # # #         RETURN coalesce(p.doi, '') AS paper_id,
# # # #                coalesce(p.title, 'Untitled') AS title,
# # # #                coalesce(p.year, 0) AS year,
# # # #                coalesce(p.doi, '') AS doi_link,
# # # #                [] AS authors,
# # # #                coalesce(r.name, '') AS researcher
# # # #         ORDER BY coalesce(p.year, 0) DESC
# # # #         LIMIT $k
# # # #         """
# # # #     else:
# # # #         cypher = """
# # # #         MATCH (r:Researcher {name: $name})-[:WROTE]->(p:Paper)
# # # #         OPTIONAL MATCH (p)-[:HAS_AUTHOR]->(a:Author)
# # # #         RETURN coalesce(p.paper_id, '') AS paper_id,
# # # #                coalesce(p.title, 'Untitled') AS title,
# # # #                coalesce(p.year, 0) AS year,
# # # #                coalesce(p.doi_link, '') AS doi_link,
# # # #                collect(DISTINCT a.name)[..20] AS authors,
# # # #                coalesce(r.name, '') AS researcher
# # # #         ORDER BY coalesce(p.year, 0) DESC
# # # #         LIMIT $k
# # # #         """
    
# # # #     return _run_query(cypher, name=name, k=k)

# # # """
# # # graph_retriever.py - Fully dynamic Neo4j queries using ChromaDB content
# # # Works with existing database - no re-ingestion needed
# # # """
# # # from typing import List, Dict, Optional
# # # import re
# # # from neo4j import GraphDatabase
# # # from database_manager import get_active_db_config

# # # _driver = None


# # # def get_neo4j_driver():
# # #     """Get Neo4j driver from active database config."""
# # #     global _driver
# # #     config = get_active_db_config()
    
# # #     if _driver is None:
# # #         _driver = GraphDatabase.driver(
# # #             config.neo4j_uri,
# # #             auth=(config.neo4j_user, config.neo4j_password)
# # #         )
    
# # #     return _driver


# # # def close_driver():
# # #     """Close Neo4j driver."""
# # #     global _driver
# # #     if _driver:
# # #         _driver.close()
# # #         _driver = None


# # # def _safe_str(v):
# # #     return str(v).strip() if v else ""


# # # def _safe_int(v):
# # #     try:
# # #         return int(v) if v else 0
# # #     except (ValueError, TypeError):
# # #         return 0


# # # def _normalize(txt: str) -> str:
# # #     return re.sub(r"[^a-z0-9 ]+", "", _safe_str(txt).lower())


# # # def _similarity(a, b):
# # #     a, b = set(_normalize(a).split()), set(_normalize(b).split())
# # #     if not a or not b:
# # #         return 0
# # #     return len(a & b) / len(a | b)


# # # def _run_query(cypher: str, **params) -> List[Dict]:
# # #     """Execute Cypher query using active database config."""
# # #     try:
# # #         config = get_active_db_config()
# # #         driver = get_neo4j_driver()
        
# # #         with driver.session(database=config.neo4j_database) as s:
# # #             return [dict(r) for r in s.run(cypher, **params)]
# # #     except Exception as e:
# # #         print(f"❌ Neo4j query error: {e}")
# # #         return []


# # # def extract_names_from_text(text: str) -> List[str]:
# # #     """
# # #     Extract potential researcher/author names from text.
# # #     Looks for capitalized word patterns that could be names.
# # #     """
# # #     if not text:
# # #         return []
    
# # #     # Pattern: Capitalized words (2-4 words that could be a name)
# # #     # Example: "John Smith", "Mary Jane Watson"
# # #     name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
# # #     potential_names = re.findall(name_pattern, text)
    
# # #     # Also look for semicolon/comma separated names
# # #     separator_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?:\s*[;,]\s*)'
# # #     separated_names = re.findall(separator_pattern, text)
    
# # #     all_names = list(set(potential_names + separated_names))
    
# # #     # Filter out common false positives
# # #     false_positives = {'The', 'This', 'That', 'These', 'Those', 'Paper', 'Abstract', 'Title'}
# # #     filtered_names = [n for n in all_names if n not in false_positives and len(n) > 3]
    
# # #     return filtered_names[:20]  # Limit to top 20


# # # def extract_search_terms_from_chroma_doc(doc_content: str, metadata: Dict) -> Dict[str, List[str]]:
# # #     """
# # #     Extract search terms from ChromaDB document content AND metadata.
# # #     Works with existing database structure.
# # #     """
# # #     search_terms = {
# # #         "titles": [],
# # #         "dois": [],
# # #         "names": [],
# # #         "keywords": []
# # #     }
    
# # #     # Extract from metadata
# # #     for key, value in metadata.items():
# # #         value_str = _safe_str(value)
# # #         if not value_str:
# # #             continue
        
# # #         # Titles
# # #         if 'title' in key.lower():
# # #             search_terms["titles"].append(value_str)
        
# # #         # DOIs
# # #         elif 'doi' in key.lower():
# # #             search_terms["dois"].append(value_str)
        
# # #         # Names (researcher, authors, etc.)
# # #         elif any(term in key.lower() for term in ['author', 'researcher', 'name']):
# # #             # Split on common separators
# # #             names = re.split(r'[;,\|&]', value_str)
# # #             search_terms["names"].extend([n.strip() for n in names if len(n.strip()) > 2])
    
# # #     # Extract names directly from document content
# # #     content_names = extract_names_from_text(doc_content)
# # #     search_terms["names"].extend(content_names)
    
# # #     # Extract keywords from document content
# # #     words = re.findall(r'\b[A-Za-z]{4,}\b', doc_content)
# # #     # Remove common stop words
# # #     stop_words = {'abstract', 'paper', 'research', 'study', 'university', 'that', 'this', 'with', 'from'}
# # #     keywords = [w for w in words if w.lower() not in stop_words]
# # #     search_terms["keywords"] = list(set(keywords))[:15]
    
# # #     # Deduplicate names
# # #     search_terms["names"] = list(set(search_terms["names"]))
    
# # #     return search_terms


# # # def search_neo4j_dynamically(search_terms: Dict, limit: int = 25) -> List[Dict]:
# # #     """
# # #     Search Neo4j using extracted terms with multiple strategies.
# # #     Completely dynamic - no hardcoded field names!
# # #     """
# # #     all_results = []
# # #     seen_ids = set()
    
# # #     # Strategy 1: Search by DOI (most precise)
# # #     if search_terms["dois"]:
# # #         cypher = """
# # #         MATCH (p:Paper)
# # #         WHERE ANY(doi_val IN $dois WHERE 
# # #             ANY(prop IN keys(p) WHERE toLower(toString(p[prop])) CONTAINS toLower(doi_val)))
# # #         OPTIONAL MATCH (p)<-[r]-(person)
# # #         WHERE person.name IS NOT NULL
# # #         RETURN p, collect(DISTINCT person.name) AS related_names
# # #         LIMIT $limit
# # #         """
# # #         results = _run_query(cypher, dois=search_terms["dois"], limit=limit)
        
# # #         for r in results:
# # #             paper = r.get('p')
# # #             if paper:
# # #                 node_id = paper.id
# # #                 if node_id not in seen_ids:
# # #                     seen_ids.add(node_id)
# # #                     all_results.append(_extract_paper_info(paper, r.get('related_names', [])))
    
# # #     # Strategy 2: Search by title
# # #     if search_terms["titles"] and len(all_results) < limit:
# # #         cypher = """
# # #         MATCH (p:Paper)
# # #         WHERE ANY(title_val IN $titles WHERE 
# # #             ANY(prop IN keys(p) WHERE 
# # #                 prop =~ '(?i).*title.*' AND 
# # #                 toLower(toString(p[prop])) CONTAINS toLower(title_val)))
# # #         OPTIONAL MATCH (p)<-[r]-(person)
# # #         WHERE person.name IS NOT NULL
# # #         RETURN p, collect(DISTINCT person.name) AS related_names
# # #         LIMIT $limit
# # #         """
# # #         results = _run_query(cypher, titles=search_terms["titles"], limit=limit)
        
# # #         for r in results:
# # #             paper = r.get('p')
# # #             if paper:
# # #                 node_id = paper.id
# # #                 if node_id not in seen_ids:
# # #                     seen_ids.add(node_id)
# # #                     all_results.append(_extract_paper_info(paper, r.get('related_names', [])))
    
# # #     # Strategy 3: Search by names (researchers/authors)
# # #     if search_terms["names"] and len(all_results) < limit:
# # #         cypher = """
# # #         UNWIND $names AS person_name
# # #         MATCH (person)
# # #         WHERE person.name IS NOT NULL AND toLower(person.name) CONTAINS toLower(person_name)
# # #         MATCH (person)-[r]->(p:Paper)
# # #         OPTIONAL MATCH (p)<-[r2]-(other)
# # #         WHERE other.name IS NOT NULL
# # #         RETURN DISTINCT p, collect(DISTINCT other.name) AS related_names
# # #         LIMIT $limit
# # #         """
# # #         # Use top 10 names to avoid query overload
# # #         top_names = search_terms["names"][:10]
# # #         results = _run_query(cypher, names=top_names, limit=limit)
        
# # #         for r in results:
# # #             paper = r.get('p')
# # #             if paper:
# # #                 node_id = paper.id
# # #                 if node_id not in seen_ids:
# # #                     seen_ids.add(node_id)
# # #                     all_results.append(_extract_paper_info(paper, r.get('related_names', [])))
    
# # #     # Strategy 4: Keyword search (fallback)
# # #     if search_terms["keywords"] and len(all_results) < limit:
# # #         cypher = """
# # #         UNWIND $keywords AS kw
# # #         MATCH (p:Paper)
# # #         WHERE ANY(prop IN keys(p) WHERE toLower(toString(p[prop])) CONTAINS toLower(kw))
# # #         OPTIONAL MATCH (p)<-[r]-(person)
# # #         WHERE person.name IS NOT NULL
# # #         RETURN DISTINCT p, collect(DISTINCT person.name) AS related_names
# # #         LIMIT $limit
# # #         """
# # #         top_keywords = sorted(set(search_terms["keywords"]), key=lambda x: len(x), reverse=True)[:5]
# # #         results = _run_query(cypher, keywords=top_keywords, limit=limit)
        
# # #         for r in results:
# # #             paper = r.get('p')
# # #             if paper:
# # #                 node_id = paper.id
# # #                 if node_id not in seen_ids:
# # #                     seen_ids.add(node_id)
# # #                     all_results.append(_extract_paper_info(paper, r.get('related_names', [])))
    
# # #     return all_results[:limit]


# # # def _extract_paper_info(paper_node, related_names: List[str]) -> Dict:
# # #     """
# # #     Dynamically extract paper information from Neo4j node.
# # #     Works with any property names!
# # #     """
# # #     info = {}
    
# # #     # Extract all properties dynamically
# # #     for key, value in paper_node.items():
# # #         info[key] = value
    
# # #     # Try to find standard fields by pattern matching
# # #     paper_id = None
# # #     title = None
# # #     year = None
# # #     doi = None
    
# # #     for key, value in paper_node.items():
# # #         key_lower = key.lower()
        
# # #         if not paper_id and any(term in key_lower for term in ['paper_id', 'id', 'doi']):
# # #             paper_id = _safe_str(value)
        
# # #         if not title and 'title' in key_lower:
# # #             title = _safe_str(value)
        
# # #         if not year and 'year' in key_lower:
# # #             year = _safe_int(value)
        
# # #         if not doi and 'doi' in key_lower:
# # #             doi = _safe_str(value)
    
# # #     # Set standardized fields
# # #     info['paper_id'] = paper_id or str(paper_node.id)
# # #     info['title'] = title or 'Untitled'
# # #     info['year'] = year or 0
# # #     info['doi'] = doi or ''
# # #     info['authors'] = related_names if related_names else []
# # #     info['researcher'] = related_names[0] if related_names else 'Unknown'
# # #     info['related'] = []
# # #     info['info'] = ''
# # #     info['source'] = ''
# # #     info['score'] = 0.0
    
# # #     return info


# # # def weighted_search(query_text: str, candidates):
# # #     """Rank papers with recency bias."""
# # #     q_norm = _normalize(query_text)
# # #     out = []
# # #     current_year = 2025
    
# # #     for row in candidates:
# # #         score = (
# # #             0.6 * _similarity(q_norm, row.get("researcher", "")) +
# # #             0.25 * _similarity(q_norm, row.get("title", "")) +
# # #             0.1 * sum(_similarity(q_norm, a) for a in row.get("authors", [])) / max(len(row.get("authors", [])), 1) +
# # #             0.05 * sum(_similarity(q_norm, r) for r in row.get("related", [])) / max(len(row.get("related", [])), 1)
# # #         )
        
# # #         year = _safe_int(row.get("year", 0))
# # #         if year > 0:
# # #             recency_bonus = max(0, (year - (current_year - 5)) / 5 * 0.2)
# # #             score += recency_bonus
        
# # #         row["score"] = round(score, 3)
# # #         row["year"] = year
# # #         out.append(row)
    
# # #     out.sort(key=lambda x: (-x["score"], -x.get("year", 0)))
# # #     return out


# # # def search_graph_from_chroma_meta_enhanced(query_text, chroma_docs_with_meta, k=8):
# # #     """
# # #     Use ChromaDB document content + metadata to search Neo4j.
# # #     Works with existing database!
# # #     """
# # #     all_rows = []
# # #     seen_papers = set()
    
# # #     for doc, meta in chroma_docs_with_meta:
# # #         # Extract search terms from BOTH content and metadata
# # #         search_terms = extract_search_terms_from_chroma_doc(doc.page_content, meta)
        
# # #         # Search Neo4j dynamically
# # #         rows = search_neo4j_dynamically(search_terms, limit=10)
        
# # #         for row in rows:
# # #             pid = row.get("paper_id", "")
# # #             if pid and pid not in seen_papers:
# # #                 seen_papers.add(pid)
# # #                 all_rows.append(row)
    
# # #     ranked = weighted_search(query_text, all_rows)
# # #     return ranked[:k]


# # # def query_graph(question: str, k: int = 10) -> List[Dict]:
# # #     """Direct graph query using dynamic keyword extraction."""
# # #     keywords = re.findall(r'\b[A-Za-z]{3,}\b', question)
    
# # #     cypher = """
# # #     UNWIND $keywords AS kw
# # #     MATCH (p:Paper)
# # #     WHERE ANY(prop IN keys(p) WHERE toLower(toString(p[prop])) CONTAINS toLower(kw))
# # #     OPTIONAL MATCH (p)<-[r]-(person)
# # #     WHERE person.name IS NOT NULL
# # #     WITH DISTINCT p, collect(DISTINCT person.name) AS related_names
# # #     RETURN p, related_names
# # #     ORDER BY CASE WHEN p.year IS NOT NULL THEN toInteger(p.year) ELSE 0 END DESC
# # #     LIMIT $limit
# # #     """
    
# # #     results = _run_query(cypher, keywords=keywords[:10], limit=max(k*2, 20))
    
# # #     papers = []
# # #     for r in results:
# # #         paper = r.get('p')
# # #         if paper:
# # #             papers.append(_extract_paper_info(paper, r.get('related_names', [])))
    
# # #     return papers[:k]


# # # def get_papers_by_researcher(name: str, k: int = 25) -> List[Dict]:
# # #     """Dynamic researcher query."""
# # #     cypher = """
# # #     MATCH (person)
# # #     WHERE person.name IS NOT NULL AND toLower(person.name) CONTAINS toLower($name)
# # #     MATCH (person)-[r]->(p:Paper)
# # #     OPTIONAL MATCH (p)<-[r2]-(other)
# # #     WHERE other.name IS NOT NULL
# # #     RETURN DISTINCT p, collect(DISTINCT other.name) AS related_names
# # #     ORDER BY CASE WHEN p.year IS NOT NULL THEN toInteger(p.year) ELSE 0 END DESC
# # #     LIMIT $k
# # #     """
    
# # #     results = _run_query(cypher, name=name, k=k)
    
# # #     papers = []
# # #     for r in results:
# # #         paper = r.get('p')
# # #         if paper:
# # #             papers.append(_extract_paper_info(paper, r.get('related_names', [])))
    
# # #     return papers

# # """
# # graph_retriever.py - Fully semantic, no hardcoded field matching
# # """
# # from typing import List, Dict, Optional
# # import re
# # from neo4j import GraphDatabase
# # from database_manager import get_active_db_config

# # _driver = None


# # def get_neo4j_driver():
# #     global _driver
# #     config = get_active_db_config()
    
# #     if _driver is None:
# #         _driver = GraphDatabase.driver(
# #             config.neo4j_uri,
# #             auth=(config.neo4j_user, config.neo4j_password)
# #         )
    
# #     return _driver


# # def close_driver():
# #     global _driver
# #     if _driver:
# #         _driver.close()
# #         _driver = None


# # def _safe_str(v):
# #     return str(v).strip() if v else ""


# # def _safe_int(v):
# #     try:
# #         return int(v) if v else 0
# #     except (ValueError, TypeError):
# #         return 0


# # def _run_query(cypher: str, **params) -> List[Dict]:
# #     try:
# #         config = get_active_db_config()
# #         driver = get_neo4j_driver()
        
# #         with driver.session(database=config.neo4j_database) as s:
# #             return [dict(r) for r in s.run(cypher, **params)]
# #     except Exception as e:
# #         print(f"❌ Neo4j query error: {e}")
# #         return []


# # def query_neo4j_with_metadata(metadata_list: List[Dict], limit: int = 25) -> List[Dict]:
# #     """
# #     FULLY SEMANTIC: Query Neo4j using ANY metadata from ChromaDB.
# #     No hardcoded field names - searches ALL properties dynamically.
# #     """
# #     all_results = []
# #     seen_ids = set()
    
# #     print(f"🔍 Querying Neo4j with metadata from {len(metadata_list)} ChromaDB results")
    
# #     for meta in metadata_list:
# #         if not meta:
# #             continue
        
# #         # Extract ALL non-empty values from metadata
# #         search_values = [
# #             _safe_str(v) for v in meta.values() 
# #             if v and _safe_str(v) not in ['Unknown', 'N/A', '']
# #         ]
        
# #         if not search_values:
# #             continue
        
# #         # Semantic Neo4j query: Find papers where ANY property matches ANY metadata value
# #         cypher = """
# #         UNWIND $values AS val
# #         MATCH (p:Paper)
# #         WHERE ANY(prop IN keys(p) WHERE 
# #             toString(p[prop]) = val OR
# #             toLower(toString(p[prop])) = toLower(val) OR
# #             toLower(toString(p[prop])) CONTAINS toLower(val))
# #         WITH DISTINCT p
# #         OPTIONAL MATCH (p)<-[r]-(person)
# #         WHERE person.name IS NOT NULL
# #         RETURN p, collect(DISTINCT person.name) AS related_names
# #         LIMIT $limit
# #         """
        
# #         results = _run_query(cypher, values=search_values, limit=5)
        
# #         for r in results:
# #             paper = r.get('p')
# #             if paper:
# #                 node_id = paper.id
# #                 if node_id not in seen_ids:
# #                     seen_ids.add(node_id)
# #                     all_results.append(_extract_paper_info(paper, r.get('related_names', [])))
    
# #     print(f"✅ Neo4j semantic search found {len(all_results)} unique papers")
    
# #     # Sort by year (recent first)
# #     all_results.sort(key=lambda x: -x.get('year', 0))
    
# #     return all_results[:limit]


# # def _extract_paper_info(paper_node, related_names: List[str]) -> Dict:
# #     """Dynamically extract ALL properties from Neo4j node."""
# #     info = {
# #         'paper_id': str(paper_node.id),
# #         'title': 'Untitled',
# #         'year': 0,
# #         'doi': '',
# #         'authors': related_names if related_names else [],
# #         'researcher': related_names[0] if related_names else 'Unknown',
# #         'score': 0.0
# #     }
    
# #     # Extract ALL properties dynamically
# #     for key, value in paper_node.items():
# #         key_lower = key.lower()
        
# #         # Update standard fields if found
# #         if 'title' in key_lower and not info['title'] != 'Untitled':
# #             info['title'] = _safe_str(value)
        
# #         if 'year' in key_lower:
# #             info['year'] = _safe_int(value)
        
# #         if 'doi' in key_lower and not info['doi']:
# #             info['doi'] = _safe_str(value)
        
# #         if any(term in key_lower for term in ['paper_id', 'id']) and info['paper_id'] == str(paper_node.id):
# #             info['paper_id'] = _safe_str(value)
        
# #         # Keep original property
# #         info[key] = value
    
# #     return info


# # def detect_person_name_in_query(query: str) -> Optional[str]:
# #     """Simple capitalized word detection - no hardcoded patterns."""
# #     words = re.findall(r'\b([A-Z][a-z]+)\b', query)
    
# #     # Find sequences of capitalized words (likely names)
# #     capitalized_sequences = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', query)
    
# #     return capitalized_sequences[0] if capitalized_sequences else None


# # def search_neo4j_by_person_name(person_name: str, limit: int = 25) -> List[Dict]:
# #     """Semantic person search in Neo4j."""
# #     cypher = """
# #     MATCH (person)
# #     WHERE person.name IS NOT NULL AND 
# #           toLower(person.name) CONTAINS toLower($name)
# #     MATCH (person)-[r]->(p:Paper)
# #     OPTIONAL MATCH (p)<-[r2]-(other)
# #     WHERE other.name IS NOT NULL
# #     RETURN DISTINCT p, person.name AS primary_researcher, collect(DISTINCT other.name) AS related_names
# #     ORDER BY CASE WHEN p.year IS NOT NULL THEN toInteger(p.year) ELSE 0 END DESC
# #     LIMIT $limit
# #     """
    
# #     results = _run_query(cypher, name=person_name, limit=limit)
    
# #     papers = []
# #     for r in results:
# #         paper = r.get('p')
# #         if paper:
# #             info = _extract_paper_info(paper, r.get('related_names', []))
# #             info['researcher'] = r.get('primary_researcher', 'Unknown')
# #             papers.append(info)
    
# #     return papers

# """
# graph_retriever.py - Minimal, only provides utilities
# Neo4j is queried directly from graph_visualizer
# """
# from typing import List, Dict, Optional
# import re
# from neo4j import GraphDatabase
# from database_manager import get_active_db_config

# _driver = None


# def get_neo4j_driver():
#     global _driver
#     config = get_active_db_config()
    
#     if _driver is None:
#         _driver = GraphDatabase.driver(
#             config.neo4j_uri,
#             auth=(config.neo4j_user, config.neo4j_password)
#         )
    
#     return _driver


# def close_driver():
#     global _driver
#     if _driver:
#         _driver.close()
#         _driver = None


# def _safe_str(v):
#     return str(v).strip() if v else ""


# def _safe_int(v):
#     try:
#         return int(v) if v else 0
#     except (ValueError, TypeError):
#         return 0
"""
graph_retriever.py - Minimal, only provides utilities
Neo4j is queried directly from graph_visualizer
"""
from typing import List, Dict, Optional
import re
from neo4j import GraphDatabase
from database_manager import get_active_db_config

_driver = None


def get_neo4j_driver():
    global _driver
    config = get_active_db_config()
    
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )
    
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def _safe_str(v):
    return str(v).strip() if v else ""


def _safe_int(v):
    try:
        return int(v) if v else 0
    except (ValueError, TypeError):
        return 0