# # from kg_retriever import query_graph, get_papers_by_author, get_papers_by_researcher

# # def search_graph(query: str, k: int = 10):
# #     return query_graph(query, k=k)

# # def papers_by_author(name: str, k: int = 25):
# #     return get_papers_by_author(name, k=k)

# # def papers_by_researcher(name: str, k: int = 25):
# #     return get_papers_by_researcher(name, k=k)


# # graph_retriever.py
# from neo4j import GraphDatabase
# import config_full as config
# import re

# # ------------------ Neo4j setup ------------------
# driver = GraphDatabase.driver(
#     config.NEO4J_URI,
#     auth=(config.NEO4J_USER, config.NEO4J_PASS)
# )
# DB_NAME = getattr(config, "NEO4J_DB", "neo4j")

# # ------------------ helpers ------------------
# def _safe_str(v):
#     return str(v).strip() if v else ""

# def _normalize(txt: str):
#     return re.sub(r"[^a-z0-9 ]+", "", _safe_str(txt).lower())

# def _similarity(a, b):
#     a, b = set(_normalize(a).split()), set(_normalize(b).split())
#     if not a or not b:
#         return 0
#     return len(a & b) / len(a | b)

# # ------------------ main graph search ------------------
# def search_graph(query_text: str, k: int = 5):
#     """
#     Retrieve related graph nodes based on fuzzy match against title, researcher, or authors.
#     Expands context with RELATED_TO / CITED_BY links.
#     """
#     q_norm = _normalize(query_text)
#     if not q_norm:
#         return []

#     cypher = """
#     MATCH (p:Paper)
#     OPTIONAL MATCH (p)-[:AUTHORED_BY]->(r:Researcher)
#     OPTIONAL MATCH (p)-[:CITED_BY|:RELATED_TO]->(related)
#     RETURN p.title AS title,
#            p.year AS year,
#            p.doi AS doi,
#            p.info AS info,
#            r.name AS researcher,
#            COLLECT(DISTINCT r.name) AS authors,
#            COLLECT(DISTINCT related.title) AS related_papers
#     """

#     results = []
#     try:
#         with driver.session(database=DB_NAME) as session:
#             data = session.run(cypher)
#             for row in data:
#                 title = _safe_str(row["title"])
#                 researcher = _safe_str(row["researcher"])
#                 authors = row["authors"] or []
#                 info = _safe_str(row["info"])
#                 doi = _safe_str(row["doi"])
#                 year = _safe_str(row["year"])
#                 related = row["related_papers"] or []

#                 score = max(
#                     _similarity(q_norm, title),
#                     _similarity(q_norm, researcher),
#                     *[_similarity(q_norm, a) for a in authors]
#                 )

#                 if score > 0.1:
#                     results.append({
#                         "title": title,
#                         "researcher": researcher,
#                         "authors": authors,
#                         "year": year,
#                         "doi": doi,
#                         "info": info,
#                         "related": related,
#                         "score": score
#                     })
#         results.sort(key=lambda x: x["score"], reverse=True)
#         return results[:k]

#     except Exception as e:
#         print(f"❌ Neo4j query error: {e}")
#         return []

# if __name__ == "__main__":
#     q = "Jeffrey Saltz"
#     hits = search_graph(q, k=10)
#     for h in hits:
#         print(f"{h['score']:.2f} | {h['title']} — {h['researcher']}")

# graph_retriever.py
from neo4j import GraphDatabase
import config_full as config
import re

driver = GraphDatabase.driver(
    config.NEO4J_URI,
    auth=(config.NEO4J_USER, config.NEO4J_PASS)
)
DB_NAME = getattr(config, "NEO4J_DB", "neo4j")

# ------------------ helpers ------------------
def _safe_str(v): return str(v).strip() if v else ""
def _normalize(txt: str): return re.sub(r"[^a-z0-9 ]+", "", _safe_str(txt).lower())
def _similarity(a, b):
    a, b = set(_normalize(a).split()), set(_normalize(b).split())
    if not a or not b: return 0
    return len(a & b) / len(a | b)

# ------------------ graph retrieval ------------------
def expand_graph_context(researcher=None, title=None, doi=None, hops: int = 1):
    """
    Pull a small subgraph around the node(s) identified by researcher/title/doi.
    """
    match_clause = []
    if researcher:
        match_clause.append(f"(r:Researcher {{name: '{researcher}'}})-[:WROTE]->(p:Paper)")
    elif title:
        match_clause.append(f"(p:Paper) WHERE toLower(p.title) CONTAINS toLower('{title}')")
    elif doi:
        match_clause.append(f"(p:Paper {{doi: '{doi}'}})")
    else:
        return []

    match_block = "MATCH " + " ".join(match_clause)
    rels = "OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)\nOPTIONAL MATCH (p)-[:RELATED_TO|:CITED_BY]->(r2:Paper)"
    cypher = f"""
    {match_block}
    {rels}
    RETURN DISTINCT p.title AS title,
                    p.year AS year,
                    p.doi AS doi,
                    collect(DISTINCT a.name) AS authors,
                    collect(DISTINCT r2.title) AS related,
                    p.info AS info,
                    '{researcher or ''}' AS researcher
    LIMIT 25
    """

    try:
        with driver.session(database=DB_NAME) as s:
            rows = [dict(r) for r in s.run(cypher)]
        return rows
    except Exception as e:
        print("❌ Graph expansion error:", e)
        return []

def weighted_search(query_text: str, candidates, w_r=0.6, w_t=0.25, w_a=0.1, w_rel=0.05):
    q_norm = _normalize(query_text)
    out = []
    for row in candidates:
        score = (
            w_r * _similarity(q_norm, row.get("researcher", "")) +
            w_t * _similarity(q_norm, row.get("title", "")) +
            w_a * sum(_similarity(q_norm, a) for a in row.get("authors", [])) / (len(row.get("authors", [])) or 1) +
            w_rel * sum(_similarity(q_norm, r) for r in row.get("related", [])) / (len(row.get("related", [])) or 1)
        )
        row["score"] = round(score, 3)
        out.append(row)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def search_graph_from_chroma_meta(query_text, chroma_metas, k=8):
    """
    Restrict Neo4j exploration to nodes surfaced by Chroma metadata.
    """
    all_rows = []
    for meta in chroma_metas:
        rname = _safe_str(meta.get("researcher"))
        title = _safe_str(meta.get("title"))
        doi = _safe_str(meta.get("doi"))
        rows = expand_graph_context(rname, title, doi)
        all_rows.extend(rows)
    ranked = weighted_search(query_text, all_rows)
    return ranked[:k]
