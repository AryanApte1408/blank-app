from kg_retriever import query_graph, get_papers_by_author, get_papers_by_researcher

def search_graph(query: str, k: int = 10):
    return query_graph(query, k=k)

def papers_by_author(name: str, k: int = 25):
    return get_papers_by_author(name, k=k)

def papers_by_researcher(name: str, k: int = 25):
    return get_papers_by_researcher(name, k=k)
