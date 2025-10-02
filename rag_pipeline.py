# # # rag_pipeline.py
# # from langchain.prompts import PromptTemplate
# # from transformers import pipeline
# # import config_full as config
# # from hybrid_langchain_retriever import hybrid_retrieve

# # # Load LLaMA model (ensure Transformers format, not GGUF)
# # llm = pipeline(
# #     "text-generation",
# #     model=config.LLAMA_MODEL_PATH,
# #     device_map="auto",
# #     max_new_tokens=300,
# #     temperature=0.2,
# #     top_p=0.9
# # )

# # prompt_template = """You are a Syracuse University research assistant.

# # User question: {question}

# # Retrieved context:
# # {context}

# # Answer clearly using only the retrieved context. 
# # If the context does not contain the answer, say "Not found in retrieved materials."

# # Answer:"""

# # prompt = PromptTemplate.from_template(prompt_template)

# # def answer_question(question: str, k: int = 5):
# #     ctx = hybrid_retrieve(question, k)
# #     context = "\n---\n".join(ctx)
# #     prompt_text = prompt.format(question=question, context=context)
# #     gen = llm(prompt_text)[0]["generated_text"]
# #     return gen.split("Answer:", 1)[-1].strip()


# # rag_pipeline.py
# import re, torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import config_full as config
# from hybrid_langchain_retriever import hybrid_search

# device = "cuda" if torch.cuda.is_available() else "cpu"
# dtype = torch.float16 if device == "cuda" else torch.float32

# tok, model = None, None

# def _load_llm():
#     global tok, model
#     if tok: return tok, model
#     tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
#     model = AutoModelForCausalLM.from_pretrained(
#         config.LLAMA_MODEL_PATH,
#         torch_dtype=dtype,
#         device_map="auto" if device == "cuda" else None,
#     )
#     if tok.pad_token_id is None and tok.eos_token_id:
#         tok.pad_token_id = tok.eos_token_id
#     return tok, model

# def sanitize(blocks, max_chars=6000):
#     out, total = [], 0
#     for b in blocks:
#         b = re.sub(r"<[^>]+>", " ", b)
#         if 20 < len(b) < 500:
#             if total + len(b) > max_chars:
#                 break
#             out.append(b)
#             total += len(b)
#     return out

# def build_prompt(question, context):
#     return (
#         f"<|system|>Use only the provided context to answer.<|/system|>\n"
#         f"<|user|>\nQ: {question}\n\nContext:\n" + "\n".join(context) +
#         "\n</|user|>\n<|assistant|>"
#     )

# def answer_question(question: str, n_ctx: int = 6):
#     retr = hybrid_search(question, k_graph=n_ctx, k_chroma=n_ctx)
#     ctx = sanitize(retr["fused_text_blocks"])
#     if not ctx:
#         return {"answer": "Not found in retrieved materials."}
#     prompt = build_prompt(question, ctx)

#     tok, model = _load_llm()
#     inputs = tok(prompt, return_tensors="pt").to(model.device)
#     gen = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9)
#     out = tok.decode(gen[0], skip_special_tokens=True)
#     ans = out.split("<|assistant|>")[-1].strip()

#     return {"answer": ans, **retr}


# rag_pipeline.py
import re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config_full as config
from hybrid_langchain_retriever import hybrid_search

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tok, model = None, None

def _load_llm():
    global tok, model
    if tok: return tok, model
    tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        config.LLAMA_MODEL_PATH,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if tok.pad_token_id is None and tok.eos_token_id:
        tok.pad_token_id = tok.eos_token_id
    return tok, model

def sanitize(blocks, max_chars=6000):
    out, total = [], 0
    for b in blocks:
        b = re.sub(r"<[^>]+>", " ", b)
        if 20 < len(b) < 500:
            if total + len(b) > max_chars:
                break
            out.append(b)
            total += len(b)
    return out

def build_prompt(question, context):
    return (
        f"<|system|>Use only the provided context to answer.<|/system|>\n"
        f"<|user|>\nQ: {question}\n\nContext:\n" + "\n".join(context) +
        "\n</|user|>\n<|assistant|>"
    )

def answer_question(question: str, n_ctx: int = 6):
    retr = hybrid_search(question, k_graph=n_ctx, k_chroma=n_ctx)
    ctx = sanitize(retr["fused_text_blocks"])
    if not ctx:
        return {"answer": "Not found in retrieved materials."}
    prompt = build_prompt(question, ctx)

    tok, model = _load_llm()
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9)
    out = tok.decode(gen[0], skip_special_tokens=True)
    ans = out.split("<|assistant|>")[-1].strip()

    return {"answer": ans, **retr}
