# graph_visualizer.py
from streamlit_agraph import agraph, Node, Edge, Config

def render_graph(graph_hits):
    """
    Build and render an interactive graph view using streamlit_agraph.
    graph_hits: list of dicts [{title, researcher, authors, related, score}, ...]
    """

    if not graph_hits:
        return "No graph data available."

    nodes, edges, added = [], [], set()

    for g in graph_hits:
        title = g.get("title", "Untitled")
        researcher = g.get("researcher", "Unknown")
        authors = g.get("authors", []) or []
        related = g.get("related", []) or []

        # --- main nodes ---
        if researcher not in added:
            nodes.append(Node(id=researcher, label=researcher,
                              size=35, color="#1E90FF", shape="star"))
            added.add(researcher)

        if title not in added:
            nodes.append(Node(id=title, label=title,
                              size=25, color="#FFD700"))
            added.add(title)

        # --- relationships ---
        edges.append(Edge(source=researcher, target=title,
                          label="authored", color="#FFB86C"))

        for a in authors:
            if a not in added:
                nodes.append(Node(id=a, label=a,
                                  size=20, color="#BD93F9"))
                added.add(a)
            edges.append(Edge(source=a, target=title,
                              label="co-author", color="#6272A4"))

        for r in related[:5]:
            if r not in added:
                nodes.append(Node(id=r, label=r,
                                  size=18, color="#FF79C6"))
                added.add(r)
            edges.append(Edge(source=title, target=r,
                              label="related", color="#FF5555"))

    config = Config(
        width="100%",
        height=650,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#A6E22E",
        collapsible=True,
        link={'color': '#AAAAAA', 'labelProperty': 'label'}
    )

    return agraph(nodes=nodes, edges=edges, config=config)
