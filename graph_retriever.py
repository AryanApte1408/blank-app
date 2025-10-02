# # # graph_retriever.py
# # from neo4j import GraphDatabase
# # import config_full as config

# # driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # def query_graph(question: str, k: int = 5):
# #     """
# #     Simple Cypher keyword search for Neo4j Community Edition.
# #     Matches title or author names that CONTAIN the query string.
# #     """
# #     cypher = """
# #     MATCH (p:Paper)<-[:HAS_AUTHOR]-(a:Author)
# #     WHERE toLower(p.title) CONTAINS toLower($q) OR toLower(a.name) CONTAINS toLower($q)
# #     RETURN p.title AS title, p.year AS year, collect(a.name) AS authors
# #     LIMIT $k
# #     """
# #     with driver.session(database=config.NEO4J_DB) as session:
# #         result = session.run(cypher, q=question, k=k)
# #         return [
# #             f"{r['title']} ({r['year']}) | {', '.join(r['authors'])}"
# #             for r in result
# #         ]

# # graph_retriever.py
# from kg_retriever import query_graph, get_papers_by_author, get_papers_by_researcher

# def search_graph(query: str, k: int = 10):
#     return query_graph(query, k=k)

# def papers_by_author(name: str, k: int = 25):
#     return get_papers_by_author(name, k=k)

# def papers_by_researcher(name: str, k: int = 25):
#     return get_papers_by_researcher(name, k=k)


# graph_retriever.py
from kg_retriever import query_graph, get_papers_by_author, get_papers_by_researcher

def search_graph(query: str, k: int = 10):
    return query_graph(query, k=k)

def papers_by_author(name: str, k: int = 25):
    return get_papers_by_author(name, k=k)

def papers_by_researcher(name: str, k: int = 25):
    return get_papers_by_researcher(name, k=k)
