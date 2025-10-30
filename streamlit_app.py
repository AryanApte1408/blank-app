# # # # # # # #streamlit_app.py 
# # # # # # # import streamlit as st
# # # # # # # from rag_pipeline import answer_question

# # # # # # # st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
# # # # # # # st.title("📚 Syracuse Research Assistant (Hybrid RAG)")

# # # # # # # with st.sidebar:
# # # # # # #     st.subheader("⚙️ Settings")
# # # # # # #     n_ctx = st.slider("Results per subsystem", 3, 12, 6)
# # # # # # #     show_sources = st.checkbox("Show sources", True)

# # # # # # # if "messages" not in st.session_state:
# # # # # # #     st.session_state["messages"] = []

# # # # # # # for msg in st.session_state["messages"]:
# # # # # # #     with st.chat_message(msg["role"]):
# # # # # # #         st.markdown(msg["content"])

# # # # # # # if prompt := st.chat_input("Ask about Syracuse research…"):
# # # # # # #     st.session_state["messages"].append({"role": "user", "content": prompt})
# # # # # # #     with st.chat_message("user"):
# # # # # # #         st.markdown(prompt)

# # # # # # #     with st.chat_message("assistant"):
# # # # # # #         with st.spinner("Thinking…"):
# # # # # # #             out = answer_question(prompt, n_ctx=n_ctx)
# # # # # # #             st.markdown(out["answer"])
# # # # # # #             if show_sources and "fused_text_blocks" in out:
# # # # # # #                 with st.expander("📚 Sources"):
# # # # # # #                     st.write("\n\n".join(out["fused_text_blocks"][:10]))

# # # # # # #     st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})

# # # # # # # streamlit_app.py
# # # # # # import streamlit as st
# # # # # # from rag_pipeline import answer_question

# # # # # # st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
# # # # # # st.title("📚 Syracuse Research Assistant (Hybrid RAG)")

# # # # # # with st.sidebar:
# # # # # #     st.subheader("⚙️ Settings")
# # # # # #     n_ctx = st.slider("Results per subsystem", 3, 12, 6)
# # # # # #     show_sources = st.checkbox("Show sources", True)

# # # # # # if "messages" not in st.session_state:
# # # # # #     st.session_state["messages"] = []

# # # # # # for msg in st.session_state["messages"]:
# # # # # #     with st.chat_message(msg["role"]):
# # # # # #         st.markdown(msg["content"])

# # # # # # if prompt := st.chat_input("Ask about Syracuse research…"):
# # # # # #     st.session_state["messages"].append({"role": "user", "content": prompt})
# # # # # #     with st.chat_message("user"):
# # # # # #         st.markdown(prompt)

# # # # # #     with st.chat_message("assistant"):
# # # # # #         with st.spinner("Thinking…"):
# # # # # #             out = answer_question(prompt, n_ctx=n_ctx)
# # # # # #             st.markdown(out["answer"])
# # # # # #             if show_sources:
# # # # # #                 with st.expander("📚 Sources"):
# # # # # #                     for s in out.get("fused_text_blocks", [])[:12]:
# # # # # #                         st.markdown(s)

# # # # # #     st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})


# # # # # # streamlit_app.py
# # # # # import streamlit as st
# # # # # from rag_pipeline import answer_question
# # # # # from graph_visualizer import render_graph

# # # # # st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
# # # # # st.title("📚 Syracuse Research Assistant (Hybrid RAG)")

# # # # # with st.sidebar:
# # # # #     st.subheader("⚙️ Settings")
# # # # #     n_ctx = st.slider("Results per subsystem", 3, 12, 6)
# # # # #     show_sources = st.checkbox("Show retrieved sources", True)
# # # # #     show_graph = st.checkbox("Show Graph View", False)

# # # # # if "messages" not in st.session_state:
# # # # #     st.session_state["messages"] = []

# # # # # for msg in st.session_state["messages"]:
# # # # #     with st.chat_message(msg["role"]):
# # # # #         st.markdown(msg["content"])

# # # # # if prompt := st.chat_input("Ask about Syracuse research…"):
# # # # #     st.session_state["messages"].append({"role": "user", "content": prompt})
# # # # #     with st.chat_message("user"):
# # # # #         st.markdown(prompt)

# # # # #     with st.chat_message("assistant"):
# # # # #         with st.spinner("Retrieving and reasoning…"):
# # # # #             out = answer_question(prompt, n_ctx=n_ctx)

# # # # #             # --- Answer ---
# # # # #             st.markdown(out["answer"])

# # # # #             # --- Sources ---
# # # # #             if show_sources and out.get("sources"):
# # # # #                 with st.expander("📚 Retrieved Sources"):
# # # # #                     for s in out["sources"][:10]:
# # # # #                         st.markdown(s)

# # # # #             # --- Graph View ---
# # # # #             if show_graph and out.get("graph_hits"):
# # # # #                 st.markdown("### 📊 Graph View — Researcher ↔ Papers ↔ Relations")
# # # # #                 render_graph(out["graph_hits"])

# # # # #     st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})

# # # # # streamlit_app.py
# # # # import streamlit as st
# # # # from rag_pipeline import answer_question
# # # # from graph_visualizer import render_graph

# # # # st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
# # # # st.title("Syracuse Research Assistant (Hybrid RAG)")

# # # # with st.sidebar:
# # # #     st.subheader("Settings")
# # # #     n_ctx = st.slider("Results per subsystem", 3, 12, 6)
# # # #     show_sources = st.checkbox("Show retrieved sources", True)
# # # #     show_graph = st.checkbox("Show Graph Visualization", True)

# # # # if "messages" not in st.session_state:
# # # #     st.session_state["messages"] = []

# # # # for msg in st.session_state["messages"]:
# # # #     with st.chat_message(msg["role"]):
# # # #         st.markdown(msg["content"])

# # # # if prompt := st.chat_input("Ask about Syracuse research…"):
# # # #     st.session_state["messages"].append({"role": "user", "content": prompt})
# # # #     with st.chat_message("user"):
# # # #         st.markdown(prompt)

# # # #     with st.chat_message("assistant"):
# # # #         with st.spinner("Retrieving context and graph…"):
# # # #             out = answer_question(prompt, n_ctx=n_ctx)
# # # #             st.markdown(out["answer"])

# # # #             if show_sources:
# # # #                 with st.expander("Retrieved Sources"):
# # # #                     for s in out.get("sources", []):
# # # #                         st.markdown(f"- {s}")

# # # #             if show_graph:
# # # #                 st.markdown("### Graph View — Researcher ↔ Papers ↔ Authors")

# # # #                 # Determine Cypher query dynamically
# # # #                 researcher_name = ""
# # # #                 if out.get("graph_hits"):
# # # #                     researcher_name = out["graph_hits"][0].get("researcher", "")
# # # #                 else:
# # # #                     for word in prompt.split():
# # # #                         if word[0].isupper() and len(word) > 3:
# # # #                             researcher_name = word
# # # #                             break

# # # #                 if researcher_name:
# # # #                     cypher_query = (
# # # #                         f"MATCH p=(r:Researcher {{name:'{researcher_name}'}})"
# # # #                         f"-[:HAS_RESEARCHER|AUTHORED|WROTE]->(pa:Paper) "
# # # #                         f"RETURN p LIMIT 25;"
# # # #                     )
# # # #                 else:
# # # #                     cypher_query = (
# # # #                         "MATCH p=(r:Researcher)-[:HAS_RESEARCHER|AUTHORED|WROTE]->(pa:Paper) "
# # # #                         "RETURN p LIMIT 25;"
# # # #                     )

# # # #                 graph_output = render_graph(cypher_query, height=650)
# # # #                 st.write(graph_output)

# # # #     st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})

# # # # streamlit_app.py
# # # import streamlit as st
# # # from rag_pipeline import answer_question
# # # from graph_visualizer import render_graph

# # # st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
# # # st.title("Syracuse Research Assistant (Hybrid RAG)")

# # # with st.sidebar:
# # #     st.subheader("Settings")
# # #     n_ctx = st.slider("Results per subsystem", 3, 12, 6)
# # #     show_sources = st.checkbox("Show retrieved sources", True)
# # #     show_graph = st.checkbox("Show Graph Visualization", True)

# # # if "messages" not in st.session_state:
# # #     st.session_state["messages"] = []

# # # for msg in st.session_state["messages"]:
# # #     with st.chat_message(msg["role"]):
# # #         st.markdown(msg["content"])

# # # if prompt := st.chat_input("Ask about Syracuse research…"):
# # #     st.session_state["messages"].append({"role": "user", "content": prompt})
# # #     with st.chat_message("user"):
# # #         st.markdown(prompt)

# # #     with st.chat_message("assistant"):
# # #         with st.spinner("Retrieving context and graph…"):
# # #             out = answer_question(prompt, n_ctx=n_ctx)
# # #             st.markdown(out["answer"])

# # #             if show_sources:
# # #                 with st.expander("Retrieved Sources"):
# # #                     for s in out.get("sources", []):
# # #                         st.markdown(f"- {s}")

# # #             if show_graph:
# # #                 st.markdown("### Graph View — Researcher ↔ Papers ↔ Authors")

# # #                 graph_hits = out.get("graph_hits", []) or []
# # #                 pids = [h.get("paper_id") for h in graph_hits if h.get("paper_id")]
# # #                 titles = [h.get("title") for h in graph_hits if h.get("title")]

# # #                 if pids:
# # #                     # Drive visualization by paper_id exactly as retrieved
# # #                     cypher_query = """
# # #                     UNWIND $pids AS pid
# # #                     MATCH p0=(pa:Paper {paper_id: pid})
# # #                     OPTIONAL MATCH p1=(r:Researcher)-[:WROTE]->(pa)
# # #                     OPTIONAL MATCH p2=(pa)-[:HAS_RESEARCHER]->(r)
# # #                     OPTIONAL MATCH p3=(pa)-[:HAS_AUTHOR]->(a:Author)
# # #                     RETURN collect(p0)+collect(p1)+collect(p2)+collect(p3) AS paths
# # #                     """
# # #                     graph_output = render_graph(cypher_query, params={"pids": pids}, height=650)

# # #                 elif titles:
# # #                     # Fallback: match by exact title (case-insensitive)
# # #                     cypher_query = """
# # #                     UNWIND $titles AS t
# # #                     MATCH p0=(pa:Paper)
# # #                     WHERE toLower(pa.title) = toLower(t)
# # #                     OPTIONAL MATCH p1=(r:Researcher)-[:WROTE]->(pa)
# # #                     OPTIONAL MATCH p2=(pa)-[:HAS_RESEARCHER]->(r)
# # #                     OPTIONAL MATCH p3=(pa)-[:HAS_AUTHOR]->(a:Author)
# # #                     RETURN collect(p0)+collect(p1)+collect(p2)+collect(p3) AS paths
# # #                     """
# # #                     graph_output = render_graph(cypher_query, params={"titles": titles}, height=650)

# # #                 else:
# # #                     # Last resort: small generic view
# # #                     researcher_name = ""
# # #                     if graph_hits:
# # #                         researcher_name = graph_hits[0].get("researcher", "")
# # #                     if not researcher_name:
# # #                         for word in prompt.split():
# # #                             if word[0].isupper() and len(word) > 3:
# # #                                 researcher_name = word
# # #                                 break

# # #                     if researcher_name:
# # #                         cypher_query = (
# # #                             "MATCH p=(r:Researcher {name:$name})-[:HAS_RESEARCHER|AUTHORED|WROTE]->(pa:Paper) "
# # #                             "OPTIONAL MATCH pauth=(pa)-[:HAS_AUTHOR]->(a:Author) "
# # #                             "RETURN p, collect(pauth) AS author_paths "
# # #                             "LIMIT 60"
# # #                         )
# # #                         graph_output = render_graph(cypher_query, params={"name": researcher_name}, height=650)
# # #                     else:
# # #                         cypher_query = (
# # #                             "MATCH p=(r:Researcher)-[:HAS_RESEARCHER|AUTHORED|WROTE]->(pa:Paper) "
# # #                             "RETURN p LIMIT 25"
# # #                         )
# # #                         graph_output = render_graph(cypher_query, height=650)

# # #                 st.write(graph_output)

# # #     st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})


# # # streamlit_app.py
# # import streamlit as st
# # from rag_pipeline import answer_question
# # from graph_visualizer import render_graph_from_hits  # updated import

# # st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
# # st.title("Syracuse Research Assistant (Hybrid RAG)")

# # with st.sidebar:
# #     st.subheader("Settings")
# #     n_ctx = st.slider("Results per subsystem", 3, 12, 6)
# #     show_sources = st.checkbox("Show retrieved sources", True)
# #     show_graph = st.checkbox("Show Graph Visualization", True)

# # if "messages" not in st.session_state:
# #     st.session_state["messages"] = []

# # for msg in st.session_state["messages"]:
# #     with st.chat_message(msg["role"]):
# #         st.markdown(msg["content"])

# # if prompt := st.chat_input("Ask about Syracuse research…"):
# #     st.session_state["messages"].append({"role": "user", "content": prompt})
# #     with st.chat_message("user"):
# #         st.markdown(prompt)

# #     with st.chat_message("assistant"):
# #         with st.spinner("Retrieving context and graph…"):
# #             out = answer_question(prompt, n_ctx=n_ctx)
# #             st.markdown(out["answer"])

# #             # ---- Show retrieved sources ----
# #             if show_sources:
# #                 with st.expander("Retrieved Sources"):
# #                     for s in out.get("sources", []):
# #                         st.markdown(f"- {s}")

# #             # ---- Show graph from actual retrieval context ----
# #             if show_graph:
# #                 st.markdown("### Graph View — Researcher ↔ Papers ↔ Authors")

# #                 graph_hits = out.get("graph_hits", []) or []
# #                 if not graph_hits:
# #                     st.info("No graph context retrieved from Neo4j for this query.")
# #                 else:
# #                     graph_output, cypher_query, params = render_graph_from_hits(graph_hits, height=650)
# #                     st.write(graph_output)

# #                     with st.expander("Graph Query Used"):
# #                         st.code(cypher_query.strip(), language="cypher")
# #                         st.json(params)

# #     st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})

# """
# streamlit_app.py - Hotswappable database UI (NO SQLite)
# """
# import os
# from pathlib import Path
# import streamlit as st
# from rag_pipeline import (
#     answer_question, 
#     clear_cache, 
#     clear_conversation,
#     get_cache_stats,
#     get_conversation_summary,
#     conversation_memory
# )
# from graph_visualizer import render_graph_from_hits
# from database_manager import get_db_manager, get_active_db_config
# from hybrid_langchain_retriever import clear_chroma_cache
# from graph_retriever import close_driver
# import json

# st.set_page_config(
#     page_title="Syracuse Research Assistant",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.title("🎓 Syracuse University Research Assistant")
# st.caption("Hybrid RAG with hotswappable databases (ChromaDB + Neo4j)")

# # ============================================================================
# # DATABASE MANAGER
# # ============================================================================
# db_manager = get_db_manager()

# # Initialize session state for database config
# if "current_db_config" not in st.session_state:
#     st.session_state["current_db_config"] = db_manager.active_config_name

# # ============================================================================
# # SIDEBAR - DATABASE CONFIGURATION
# # ============================================================================
# with st.sidebar:
#     st.subheader("🗄️ Database Configuration")
    
#     # Database selector
#     available_configs = db_manager.list_configs()
#     current_config = st.selectbox(
#         "Select Database",
#         options=available_configs,
#         index=available_configs.index(st.session_state["current_db_config"]),
#         format_func=lambda x: f"📚 {x.upper()}" if x == "full" else f"📄 {x.upper()}",
#         help="Switch between different database configurations"
#     )
    
#     # Handle database switch
#     if current_config != st.session_state["current_db_config"]:
#         with st.spinner(f"Switching to {current_config}..."):
#             # Clear caches
#             clear_chroma_cache()
#             close_driver()
#             clear_cache()
            
#             # Switch config
#             db_manager.switch_config(current_config)
#             st.session_state["current_db_config"] = current_config
#             st.success(f"✅ Switched to {current_config}")
#             st.rerun()
    
#     # Show active database info (NO SQLite!)
#     active_config = get_active_db_config()
#     with st.expander("ℹ️ Active Database Info", expanded=False):
#         st.caption(f"**Mode:** {active_config.mode}")
#         st.caption(f"**Description:** {active_config.description}")
#         st.caption(f"**ChromaDB:** {Path(active_config.chroma_dir).name}")
#         st.caption(f"**Collection:** {active_config.chroma_collection}")
#         st.caption(f"**Neo4j Database:** {active_config.neo4j_database}")
#         st.caption(f"**Neo4j URI:** {active_config.neo4j_uri}")
    
#     # Validate current database
#     if st.button("🔍 Validate Database"):
#         with st.spinner("Validating..."):
#             validation = db_manager.validate_config(current_config)
#             if validation["valid"]:
#                 st.success("✅ All database connections valid")
#             else:
#                 st.error("❌ Validation failed:")
#                 for key, val in validation.items():
#                     if key != "valid" and not val:
#                         st.warning(f"- {key}: Failed")
    
#     st.divider()
    
#     # ========================================================================
#     # RETRIEVAL SETTINGS
#     # ========================================================================
#     st.subheader("⚙️ Retrieval Settings")
    
#     n_ctx = st.slider(
#         "Results per subsystem",
#         min_value=3,
#         max_value=12,
#         value=6,
#         help="Number of results from ChromaDB and Neo4j"
#     )
    
#     show_sources = st.checkbox("Show retrieved sources", value=True)
#     show_graph = st.checkbox("Show Graph Visualization", value=True)
    
#     st.divider()
    
#     # ========================================================================
#     # PERFORMANCE SETTINGS
#     # ========================================================================
#     st.subheader("🚀 Performance")
    
#     use_cache = st.checkbox(
#         "Enable retrieval caching", 
#         value=True,
#         help="Cache retrieval results for faster repeated queries"
#     )
    
#     use_conversation = st.checkbox(
#         "Enable conversation memory",
#         value=True,
#         help="Maintain context across multiple questions"
#     )
    
#     # Cache stats
#     if use_cache:
#         cache_stats = get_cache_stats()
#         st.caption(
#             f"📊 Cache: {cache_stats['size']}/{cache_stats['maxsize']} | "
#             f"Hit rate: {cache_stats['hit_rate']}"
#         )
    
#     # Conversation stats
#     if use_conversation:
#         conv_stats = get_conversation_summary()
#         st.caption(
#             f"💬 Memory: {conv_stats['buffer_size']} recent"
#             + (f" + {conv_stats['archived_count']} archived" if conv_stats['has_summary'] else "")
#         )
    
#     st.divider()
    
#     # ========================================================================
#     # DATABASE MODE INFO
#     # ========================================================================
#     mode = active_config.mode
    
#     if mode == "abstracts":
#         st.info(
#             "📄 **Abstracts Mode:**\n"
#             "- Paper abstracts from APIs\n"
#             "- Faster retrieval\n"
#             "- Lower memory usage\n"
#             "- Good for quick searches"
#         )
#     elif mode == "full":
#         st.info(
#             "📚 **Full Papers Mode:**\n"
#             "- Complete papers with full text\n"
#             "- Detailed answers\n"
#             "- More comprehensive\n"
#             "- Slower retrieval"
#         )
#     else:
#         st.info(f"🔧 **Custom Mode:** {active_config.description}")
    
#     st.divider()
    
#     # ========================================================================
#     # TIPS
#     # ========================================================================
#     st.caption(
#         "💡 **Tips:**\n"
#         "- Switch databases on the fly\n"
#         "- Results sorted by recency\n"
#         "- Ask follow-up questions\n"
#         "- Query by researcher/topic"
#     )
    
#     st.divider()
    
#     # ========================================================================
#     # CLEAR BUTTONS
#     # ========================================================================
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if st.button("🗑️ Clear Chat"):
#             st.session_state["messages"] = []
#             st.rerun()
    
#     with col2:
#         if st.button("🧹 Clear Cache"):
#             clear_cache()
#             clear_conversation()
#             st.success("Cache & memory cleared!")
#             st.rerun()
    
#     st.divider()
    
#     # ========================================================================
#     # ADVANCED DATABASE MANAGEMENT (NO SQLite!)
#     # ========================================================================
#     with st.expander("🔧 Advanced: Add Custom Database"):
#         st.caption("Add a custom database configuration at runtime")
        
#         with st.form("custom_db_form"):
#             custom_name = st.text_input("Config Name", value="custom_db")
#             custom_chroma_dir = st.text_input("ChromaDB Directory", value="")
#             custom_collection = st.text_input("Collection Name", value="papers_all")
#             custom_neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
#             custom_neo4j_user = st.text_input("Neo4j User", value="neo4j")
#             custom_neo4j_pass = st.text_input("Neo4j Password", value="", type="password")
#             custom_neo4j_db = st.text_input("Neo4j Database", value="neo4j")
#             custom_description = st.text_area("Description", value="Custom database")
            
#             submitted = st.form_submit_button("Add Configuration")
            
#             if submitted:
#                 if custom_name and custom_chroma_dir:
#                     try:
#                         db_manager.add_custom_config(
#                             name=custom_name,
#                             chroma_dir=custom_chroma_dir,
#                             chroma_collection=custom_collection,
#                             neo4j_uri=custom_neo4j_uri,
#                             neo4j_database=custom_neo4j_db,
#                             description=custom_description
#                         )
#                         st.success(f"✅ Added '{custom_name}' configuration")
#                         st.rerun()
#                     except Exception as e:
#                         st.error(f"❌ Error: {e}")
#                 else:
#                     st.warning("Please fill in all required fields")
    
#     # ========================================================================
#     # DEBUG PANEL
#     # ========================================================================
#     with st.expander("🔬 Debug: Conversation Memory"):
#         if use_conversation:
#             conv_context = conversation_memory.get_context()
#             if conv_context:
#                 st.text_area(
#                     "Current Memory", 
#                     conv_context, 
#                     height=150,
#                     disabled=True
#                 )
#             else:
#                 st.caption("No conversation history yet")
#         else:
#             st.caption("Conversation memory disabled")
    
#     # ========================================================================
#     # FOOTER
#     # ========================================================================
#     st.divider()
#     st.caption(f"🔄 Active: **{current_config.upper()}**")
#     st.caption(f"📊 Mode: **{mode.upper()}**")


# # ============================================================================
# # MAIN CHAT INTERFACE
# # ============================================================================

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# # Display chat messages
# for msg in st.session_state["messages"]:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
        
#         # Show metadata for assistant messages
#         if msg["role"] == "assistant" and "metadata" in msg:
#             meta = msg["metadata"]
            
#             # Create info badges
#             badges = []
#             if meta.get("cache_hit"):
#                 badges.append("⚡ Cached")
#             else:
#                 badges.append("🔍 Fresh")
            
#             if meta.get("conversation_used"):
#                 badges.append("💬 Context-aware")
            
#             badges.append(f"🗄️ {meta.get('db_config', 'unknown').upper()}")
            
#             st.caption(" | ".join(badges))
            
#             # Expandable detailed info
#             with st.expander("ℹ️ Response Details"):
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.metric("Database", meta.get('db_config', 'unknown'))
#                     st.metric("Cache Hit", "Yes" if meta.get("cache_hit") else "No")
#                 with col2:
#                     st.metric("Sources", len(meta.get('sources', [])))
#                     st.metric("Graph Hits", len(meta.get('graph_hits', [])))


# # ============================================================================
# # CHAT INPUT AND RESPONSE
# # ============================================================================

# if prompt := st.chat_input("Ask about Syracuse research…"):
#     # Add user message
#     st.session_state["messages"].append({"role": "user", "content": prompt})
    
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # Generate response
#     with st.chat_message("assistant"):
#         active_config = get_active_db_config()
#         config_emoji = "📄" if active_config.mode == "abstracts" else "📚"
        
#         with st.spinner(f"{config_emoji} Processing query with {active_config.mode} database..."):
#             try:
#                 out = answer_question(
#                     prompt, 
#                     n_ctx=n_ctx,
#                     use_cache=use_cache,
#                     use_conversation=use_conversation
#                 )
                
#                 # Display answer
#                 st.markdown(out.get("answer", "No answer generated."))
                
#                 # Show indicators
#                 badges = []
#                 if out.get("cache_hit"):
#                     badges.append("⚡ Cached")
#                 else:
#                     badges.append("🔍 Fresh retrieval")
                
#                 if out.get("conversation_used"):
#                     badges.append("💬 Context-aware")
                
#                 badges.append(f"🗄️ {out.get('db_config', 'unknown').upper()}")
                
#                 st.caption(" | ".join(badges))
                
#                 # Show sources
#                 if show_sources and out.get("sources"):
#                     with st.expander(f"📚 Retrieved Sources ({active_config.mode} mode)"):
#                         for i, s in enumerate(out["sources"], 1):
#                             st.markdown(f"**{i}.** {s}")
                
#                 # Show graph
#                 if show_graph:
#                     st.markdown("---")
#                     st.subheader(f"🕸️ Knowledge Graph ({active_config.mode} mode)")
                    
#                     graph_hits = out.get("graph_hits", [])
                    
#                     if not graph_hits:
#                         st.info(f"No graph context retrieved from {active_config.mode} database.")
#                     else:
#                         st.caption(f"Showing {len(graph_hits)} papers from Neo4j")
                        
#                         try:
#                             graph_output, cypher_query, params = render_graph_from_hits(
#                                 graph_hits,
#                                 height=650
#                             )
                            
#                             if isinstance(graph_output, str):
#                                 st.warning(graph_output)
#                             else:
#                                 st.write(graph_output)
                                
#                                 with st.expander("🔍 Graph Query Details"):
#                                     st.code(cypher_query.strip(), language="cypher")
#                                     st.json(params)
                        
#                         except Exception as viz_error:
#                             st.error(f"Graph visualization error: {viz_error}")
                
#                 # Store message with metadata
#                 st.session_state["messages"].append({
#                     "role": "assistant",
#                     "content": out["answer"],
#                     "metadata": {
#                         "cache_hit": out.get("cache_hit", False),
#                         "conversation_used": out.get("conversation_used", False),
#                         "db_config": out.get("db_config", "unknown"),
#                         "sources": out.get("sources", []),
#                         "graph_hits": out.get("graph_hits", [])
#                     }
#                 })
            
#             except Exception as e:
#                 error_msg = f"❌ Error: {str(e)}"
#                 st.error(error_msg)
#                 st.session_state["messages"].append({
#                     "role": "assistant",
#                     "content": error_msg
#                 })


# # ============================================================================
# # DATABASE EXPORT/IMPORT (Optional Advanced Feature)
# # ============================================================================
# with st.sidebar:
#     with st.expander("💾 Export/Import Configurations"):
#         st.caption("Save and load database configurations")
        
#         # Export
#         if st.button("📤 Export All Configs"):
#             configs_export = {
#                 name: config.to_dict() 
#                 for name, config in db_manager.configs.items()
#             }
#             st.download_button(
#                 label="Download configs.json",
#                 data=json.dumps(configs_export, indent=2),
#                 file_name="database_configs.json",
#                 mime="application/json"
#             )
        
#         # Import
#         uploaded_file = st.file_uploader("📥 Import Configs", type=['json'])
#         if uploaded_file:
#             try:
#                 imported_configs = json.load(uploaded_file)
#                 st.json(imported_configs)
#                 if st.button("Apply Imported Configs"):
#                     for name, cfg_dict in imported_configs.items():
#                         from database_manager import DatabaseConfig
#                         config = DatabaseConfig(**cfg_dict)
#                         db_manager.register_config(name, config)
#                     st.success(f"✅ Imported {len(imported_configs)} configurations")
#                     st.rerun()
#             except Exception as e:
#                 st.error(f"❌ Import failed: {e}")

"""
streamlit_app.py - Syracuse Research Assistant UI
(Updated: NO results-per-subsystem slider; RAG decides context size)
"""

import os
from pathlib import Path
import streamlit as st
import json

from rag_pipeline import (
    answer_question,
    clear_cache,
    clear_conversation,
    get_cache_stats,
    get_conversation_summary,
    conversation_memory,
)
from graph_visualizer import render_graph_from_hits
from database_manager import get_db_manager, get_active_db_config
from hybrid_langchain_retriever import clear_chroma_cache
from graph_retriever import close_driver

st.set_page_config(
    page_title="Syracuse University Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎓 Syracuse University Research Assistant")
st.caption("Hybrid RAG with hotswappable databases (ChromaDB + Neo4j)")


# ============================================================================
# DATABASE MANAGER
# ============================================================================
db_manager = get_db_manager()

if "current_db_config" not in st.session_state:
    st.session_state["current_db_config"] = db_manager.active_config_name


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.subheader("🗄️ Database Configuration")

    available_configs = db_manager.list_configs()
    current_config = st.selectbox(
        "Select Database",
        options=available_configs,
        index=available_configs.index(st.session_state["current_db_config"]),
        format_func=lambda x: f"📚 {x.upper()}" if x == "full" else f"📄 {x.upper()}",
        help="Switch between different database configurations",
    )

    if current_config != st.session_state["current_db_config"]:
        with st.spinner(f"Switching to {current_config}..."):
            clear_chroma_cache()
            close_driver()
            clear_cache()
            db_manager.switch_config(current_config)
            st.session_state["current_db_config"] = current_config
            st.success(f"✅ Switched to {current_config}")
            st.rerun()

    active_config = get_active_db_config()
    with st.expander("ℹ️ Active Database Info", expanded=False):
        st.caption(f"**Mode:** {active_config.mode}")
        st.caption(f"**Description:** {active_config.description}")
        st.caption(f"**ChromaDB:** {Path(active_config.chroma_dir).name}")
        st.caption(f"**Collection:** {active_config.chroma_collection}")
        st.caption(f"**Neo4j Database:** {active_config.neo4j_database}")
        st.caption(f"**Neo4j URI:** {active_config.neo4j_uri}")

    if st.button("🔍 Validate Database"):
        with st.spinner("Validating..."):
            validation = db_manager.validate_config(current_config)
            if validation.get("valid"):
                st.success("✅ All database connections valid")
            else:
                st.error("❌ Validation failed:")
                for key, val in validation.items():
                    if key != "valid" and not val:
                        st.warning(f"- {key}: Failed")

    st.divider()

    # =========================================================================
    # RETRIEVAL SETTINGS (NO SLIDER)
    # =========================================================================
    st.subheader("⚙️ Retrieval Settings")
    show_sources = st.checkbox("Show retrieved sources", value=True)
    show_graph = st.checkbox("Show Graph Visualization", value=True)

    st.divider()

    # =========================================================================
    # PERFORMANCE
    # =========================================================================
    st.subheader("🚀 Performance")

    use_cache = st.checkbox(
        "Enable retrieval caching",
        value=True,
        help="Cache retrieval results for faster repeated queries",
    )
    use_conversation = st.checkbox(
        "Enable conversation memory",
        value=True,
        help="Maintain context across multiple questions",
    )

    if use_cache:
        cache_stats = get_cache_stats() or {}
        size = cache_stats.get("size", 0)
        maxsize = cache_stats.get("maxsize", 0)
        hit_rate = cache_stats.get("hit_rate", "0.0%")
        st.caption(f"📊 Cache: {size}/{maxsize} | Hit rate: {hit_rate}")

    if use_conversation:
        conv_stats = get_conversation_summary() or {}
        buf = conv_stats.get("buffer_size", 0)
        arch = conv_stats.get("archived_count", 0)
        has_sum = conv_stats.get("has_summary", False)
        st.caption(
            f"💬 Memory: {buf} recent"
            + (f" + {arch} archived" if has_sum else "")
        )

    st.divider()

    mode = active_config.mode
    if mode == "abstracts":
        st.info(
            "📄 **Abstracts Mode:**\n"
            "- Short academic abstracts\n"
            "- Faster retrieval\n"
            "- Good for quick searches"
        )
    elif mode == "full":
        st.info(
            "📚 **Full Papers Mode:**\n"
            "- Full text / rich metadata\n"
            "- More detailed answers\n"
            "- Slightly slower"
        )
    else:
        st.info(f"🔧 **Custom Mode:** {active_config.description}")

    st.divider()

    st.caption(
        "💡 **Tips:**\n"
        "- Databases are hotswappable\n"
        "- RAG will auto-fit context to VRAM\n"
        "- Ask follow-up questions"
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state["messages"] = []
            st.rerun()
    with col2:
        if st.button("🧹 Clear Cache"):
            clear_cache()
            clear_conversation()
            st.success("Cache & memory cleared!")
            st.rerun()

    st.divider()

    with st.expander("🔧 Advanced: Add Custom Database"):
        st.caption("Add a custom database configuration at runtime")

        with st.form("custom_db_form"):
            custom_name = st.text_input("Config Name", value="custom_db")
            custom_chroma_dir = st.text_input("ChromaDB Directory", value="")
            custom_collection = st.text_input("Collection Name", value="papers_all")
            custom_neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
            custom_neo4j_user = st.text_input("Neo4j User", value="neo4j")
            custom_neo4j_pass = st.text_input(
                "Neo4j Password", value="", type="password"
            )
            custom_neo4j_db = st.text_input("Neo4j Database", value="neo4j")
            custom_description = st.text_area(
                "Description", value="Custom database"
            )

            submitted = st.form_submit_button("Add Configuration")

            if submitted:
                if custom_name and custom_chroma_dir:
                    try:
                        db_manager.add_custom_config(
                            name=custom_name,
                            chroma_dir=custom_chroma_dir,
                            chroma_collection=custom_collection,
                            neo4j_uri=custom_neo4j_uri,
                            neo4j_user=custom_neo4j_user,
                            neo4j_password=custom_neo4j_pass,
                            neo4j_database=custom_neo4j_db,
                            description=custom_description,
                        )
                        st.success(f"✅ Added '{custom_name}' configuration")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                else:
                    st.warning("Please fill in all required fields")

    with st.expander("🔬 Debug: Conversation Memory"):
        if use_conversation:
            conv_context = conversation_memory.get_context()
            if conv_context:
                st.text_area("Current Memory", conv_context, height=150, disabled=True)
            else:
                st.caption("No conversation history yet")
        else:
            st.caption("Conversation memory disabled")

    st.divider()
    st.caption(f"🔄 Active: **{current_config.upper()}**")
    st.caption(f"📊 Mode: **{mode.upper()}**")

    # Export / Import
    with st.expander("💾 Export/Import Configurations"):
        st.caption("Save and load database configurations")

        if st.button("📤 Export All Configs"):
            configs_export = {
                name: cfg.to_dict() for name, cfg in db_manager.configs.items()
            }
            st.download_button(
                label="Download configs.json",
                data=json.dumps(configs_export, indent=2),
                file_name="database_configs.json",
                mime="application/json",
            )

        uploaded_file = st.file_uploader("📥 Import Configs", type=["json"])
        if uploaded_file:
            try:
                imported_configs = json.load(uploaded_file)
                st.json(imported_configs)
                if st.button("Apply Imported Configs"):
                    from database_manager import DatabaseConfig

                    for name, cfg_dict in imported_configs.items():
                        cfg = DatabaseConfig(**cfg_dict)
                        db_manager.register_config(name, cfg)
                    st.success(f"✅ Imported {len(imported_configs)} configurations")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Import failed: {e}")


# ============================================================================
# MAIN CHAT
# ============================================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            badges = []
            badges.append("⚡ Cached" if meta.get("cache_hit") else "🔍 Fresh")
            if meta.get("conversation_used"):
                badges.append("💬 Context-aware")
            badges.append(f"🗄️ {meta.get('db_config', 'unknown').upper()}")
            st.caption(" | ".join(badges))

            with st.expander("ℹ️ Response Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Database", meta.get("db_config", "unknown"))
                    st.metric("Cache Hit", "Yes" if meta.get("cache_hit") else "No")
                with col2:
                    st.metric("Sources", len(meta.get("sources", [])))
                    st.metric("Graph Hits", len(meta.get("graph_hits", [])))


if prompt := st.chat_input("Ask about Syracuse research…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        active_config = get_active_db_config()
        config_emoji = "📄" if active_config.mode == "abstracts" else "📚"

        with st.spinner(f"{config_emoji} Processing query with {active_config.mode} database..."):
            try:
                # n_ctx=None → let RAG decide based on VRAM
                out = answer_question(
                    prompt,
                    n_ctx=None,
                    use_cache=use_cache,
                    use_conversation=use_conversation,
                )

                st.markdown(out.get("answer", "No answer generated."))

                badges = []
                badges.append("⚡ Cached" if out.get("cache_hit") else "🔍 Fresh retrieval")
                if out.get("conversation_used"):
                    badges.append("💬 Context-aware")
                badges.append(f"🗄️ {out.get('db_config', 'unknown').upper()}")
                st.caption(" | ".join(badges))

                if show_sources and out.get("sources"):
                    with st.expander(
                        f"📚 Retrieved Sources ({active_config.mode} mode)"
                    ):
                        for i, s in enumerate(out["sources"], 1):
                            st.markdown(f"**{i}.** {s}")

                if show_graph:
                    st.markdown("---")
                    st.subheader(f"🕸️ Knowledge Graph ({active_config.mode} mode)")

                    graph_hits = out.get("graph_hits", [])
                    if not graph_hits:
                        st.info(
                            f"No graph context retrieved from {active_config.mode} database."
                        )
                    else:
                        st.caption(f"Showing {len(graph_hits)} papers from Neo4j")
                        try:
                            graph_output, cypher_query, params = render_graph_from_hits(
                                graph_hits, height=650
                            )
                            if isinstance(graph_output, str):
                                st.warning(graph_output)
                            else:
                                st.write(graph_output)
                                with st.expander("🔍 Graph Query Details"):
                                    st.code(cypher_query.strip(), language="cypher")
                                    st.json(params)
                        except Exception as viz_error:
                            st.error(f"Graph visualization error: {viz_error}")

                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": out["answer"],
                        "metadata": {
                            "cache_hit": out.get("cache_hit", False),
                            "conversation_used": out.get("conversation_used", False),
                            "db_config": out.get("db_config", "unknown"),
                            "sources": out.get("sources", []),
                            "graph_hits": out.get("graph_hits", []),
                        },
                    }
                )

            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": err}
                )
