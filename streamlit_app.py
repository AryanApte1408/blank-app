# # #streamlit_app.py 
# # import streamlit as st
# # from rag_pipeline import answer_question

# # st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
# # st.title("📚 Syracuse Research Assistant (Hybrid RAG)")

# # with st.sidebar:
# #     st.subheader("⚙️ Settings")
# #     n_ctx = st.slider("Results per subsystem", 3, 12, 6)
# #     show_sources = st.checkbox("Show sources", True)

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
# #         with st.spinner("Thinking…"):
# #             out = answer_question(prompt, n_ctx=n_ctx)
# #             st.markdown(out["answer"])
# #             if show_sources and "fused_text_blocks" in out:
# #                 with st.expander("📚 Sources"):
# #                     st.write("\n\n".join(out["fused_text_blocks"][:10]))

# #     st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})

# # streamlit_app.py
# import streamlit as st
# from rag_pipeline import answer_question

# st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
# st.title("📚 Syracuse Research Assistant (Hybrid RAG)")

# with st.sidebar:
#     st.subheader("⚙️ Settings")
#     n_ctx = st.slider("Results per subsystem", 3, 12, 6)
#     show_sources = st.checkbox("Show sources", True)

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# for msg in st.session_state["messages"]:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# if prompt := st.chat_input("Ask about Syracuse research…"):
#     st.session_state["messages"].append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking…"):
#             out = answer_question(prompt, n_ctx=n_ctx)
#             st.markdown(out["answer"])
#             if show_sources:
#                 with st.expander("📚 Sources"):
#                     for s in out.get("fused_text_blocks", [])[:12]:
#                         st.markdown(s)

#     st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})


# streamlit_app.py
import streamlit as st
from rag_pipeline import answer_question
from graph_visualizer import render_graph

st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
st.title("📚 Syracuse Research Assistant (Hybrid RAG)")

with st.sidebar:
    st.subheader("⚙️ Settings")
    n_ctx = st.slider("Results per subsystem", 3, 12, 6)
    show_sources = st.checkbox("Show retrieved sources", True)
    show_graph = st.checkbox("Show Graph View", False)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Syracuse research…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning…"):
            out = answer_question(prompt, n_ctx=n_ctx)

            # --- Answer ---
            st.markdown(out["answer"])

            # --- Sources ---
            if show_sources and out.get("sources"):
                with st.expander("📚 Retrieved Sources"):
                    for s in out["sources"][:10]:
                        st.markdown(s)

            # --- Graph View ---
            if show_graph and out.get("graph_hits"):
                st.markdown("### 📊 Graph View — Researcher ↔ Papers ↔ Relations")
                render_graph(out["graph_hits"])

    st.session_state["messages"].append({"role": "assistant", "content": out["answer"]})
