# # # # # # # # # #rag_pipeleine.py
# # # # # # # # # import re, torch
# # # # # # # # # from transformers import AutoTokenizer, AutoModelForCausalLM
# # # # # # # # # import config_full as config
# # # # # # # # # from hybrid_langchain_retriever import hybrid_search

# # # # # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # # # # dtype = torch.float16 if device == "cuda" else torch.float32

# # # # # # # # # tok, model = None, None

# # # # # # # # # def _load_llm():
# # # # # # # # #     global tok, model
# # # # # # # # #     if tok: return tok, model
# # # # # # # # #     tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
# # # # # # # # #     model = AutoModelForCausalLM.from_pretrained(
# # # # # # # # #         config.LLAMA_MODEL_PATH,
# # # # # # # # #         torch_dtype=dtype,
# # # # # # # # #         device_map="auto" if device == "cuda" else None,
# # # # # # # # #         low_cpu_mem_usage=True,
# # # # # # # # #     )
# # # # # # # # #     if tok.pad_token_id is None and tok.eos_token_id:
# # # # # # # # #         tok.pad_token_id = tok.eos_token_id
# # # # # # # # #     return tok, model

# # # # # # # # # def sanitize(blocks, max_chars=6000):
# # # # # # # # #     out, total = [], 0
# # # # # # # # #     for b in blocks:
# # # # # # # # #         b = re.sub(r"<[^>]+>", " ", b).strip()
# # # # # # # # #         if not b or b in ["N/A", "Unknown"]:
# # # # # # # # #             continue
# # # # # # # # #         if total + len(b) > max_chars:
# # # # # # # # #             break
# # # # # # # # #         out.append(b)
# # # # # # # # #         total += len(b)
# # # # # # # # #     return out

# # # # # # # # # def build_prompt(question, context):
# # # # # # # # #     return (
# # # # # # # # #         f"<|system|>Use only the provided context to answer.<|/system|>\n"
# # # # # # # # #         f"<|user|>\nQ: {question}\n\nContext:\n" + "\n".join(context) +
# # # # # # # # #         "\n</|user|>\n<|assistant|>"
# # # # # # # # #     )

# # # # # # # # # def answer_question(question: str, n_ctx: int = 6):
# # # # # # # # #     retr = hybrid_search(question, k_graph=n_ctx, k_chroma=n_ctx)
# # # # # # # # #     ctx = sanitize(retr["fused_text_blocks"])
# # # # # # # # #     if not ctx:
# # # # # # # # #         return {"answer": "Not found in retrieved materials."}
# # # # # # # # #     prompt = build_prompt(question, ctx)

# # # # # # # # #     tok, model = _load_llm()
# # # # # # # # #     inputs = tok(prompt, return_tensors="pt").to(model.device)
# # # # # # # # #     gen = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9)
# # # # # # # # #     out = tok.decode(gen[0], skip_special_tokens=True)
# # # # # # # # #     ans = out.split("<|assistant|>")[-1].strip()
# # # # # # # # #     return {"answer": ans, **retr}

# # # # # # # # # rag_pipeline.py
# # # # # # # # import re, torch
# # # # # # # # from transformers import AutoTokenizer, AutoModelForCausalLM
# # # # # # # # import config_full as config
# # # # # # # # from hybrid_langchain_retriever import hybrid_search

# # # # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # # # dtype = torch.float16 if device == "cuda" else torch.float32
# # # # # # # # tok, model = None, None

# # # # # # # # def _load_llm():
# # # # # # # #     global tok, model
# # # # # # # #     if tok: return tok, model
# # # # # # # #     tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
# # # # # # # #     model = AutoModelForCausalLM.from_pretrained(
# # # # # # # #         config.LLAMA_MODEL_PATH,
# # # # # # # #         torch_dtype=dtype,
# # # # # # # #         device_map="auto" if device == "cuda" else None,
# # # # # # # #         low_cpu_mem_usage=True,
# # # # # # # #     )
# # # # # # # #     if tok.pad_token_id is None and tok.eos_token_id:
# # # # # # # #         tok.pad_token_id = tok.eos_token_id
# # # # # # # #     return tok, model

# # # # # # # # def sanitize(blocks, max_chars=6000):
# # # # # # # #     out, total = [], 0
# # # # # # # #     for b in blocks:
# # # # # # # #         b = re.sub(r"<[^>]+>", " ", b)
# # # # # # # #         if 20 < len(b) < 500:
# # # # # # # #             if total + len(b) > max_chars:
# # # # # # # #                 break
# # # # # # # #             out.append(b)
# # # # # # # #             total += len(b)
# # # # # # # #     return out

# # # # # # # # def build_prompt(question, context):
# # # # # # # #     return (
# # # # # # # #         f"<|system|>Answer using the context only.<|/system|>\n"
# # # # # # # #         f"<|user|>\nQ: {question}\n\nContext:\n" + "\n".join(context) +
# # # # # # # #         "\n</|user|>\n<|assistant|>"
# # # # # # # #     )

# # # # # # # # def answer_question(question: str, n_ctx: int = 6):
# # # # # # # #     retr = hybrid_search(question, k_chroma=n_ctx, k_graph=n_ctx)
# # # # # # # #     ctx = sanitize(retr["fused_text_blocks"])
# # # # # # # #     if not ctx:
# # # # # # # #         return {"answer": "No relevant information found."}
# # # # # # # #     prompt = build_prompt(question, ctx)

# # # # # # # #     tok, model = _load_llm()
# # # # # # # #     inputs = tok(prompt, return_tensors="pt").to(model.device)
# # # # # # # #     gen = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9)
# # # # # # # #     out = tok.decode(gen[0], skip_special_tokens=True)
# # # # # # # #     ans = out.split("<|assistant|>")[-1].strip()
# # # # # # # #     return {"answer": ans, **retr}


# # # # # # # # rag_pipeline.py
# # # # # # # import re, torch
# # # # # # # from transformers import AutoTokenizer, AutoModelForCausalLM
# # # # # # # import config_full as config
# # # # # # # from hybrid_langchain_retriever import hybrid_search

# # # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # # dtype = torch.float16 if device == "cuda" else torch.float32

# # # # # # # tok, model = None, None

# # # # # # # def _load_llm():
# # # # # # #     """Load local LLaMA model once."""
# # # # # # #     global tok, model
# # # # # # #     if tok and model:
# # # # # # #         return tok, model
# # # # # # #     tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
# # # # # # #     model = AutoModelForCausalLM.from_pretrained(
# # # # # # #         config.LLAMA_MODEL_PATH,
# # # # # # #         torch_dtype=dtype,
# # # # # # #         device_map="auto" if device == "cuda" else None,
# # # # # # #         low_cpu_mem_usage=True,
# # # # # # #     )
# # # # # # #     if tok.pad_token_id is None and tok.eos_token_id:
# # # # # # #         tok.pad_token_id = tok.eos_token_id
# # # # # # #     return tok, model


# # # # # # # def sanitize(blocks, max_chars=6000):
# # # # # # #     """Remove tags and truncate overly long chunks."""
# # # # # # #     out, total = [], 0
# # # # # # #     for b in blocks:
# # # # # # #         b = re.sub(r"<[^>]+>", " ", b)
# # # # # # #         if 20 < len(b) < 500:
# # # # # # #             if total + len(b) > max_chars:
# # # # # # #                 break
# # # # # # #             out.append(b)
# # # # # # #             total += len(b)
# # # # # # #     return out


# # # # # # # def build_prompt(question, context):
# # # # # # #     """Format context for instruction-style LLaMA prompt."""
# # # # # # #     return (
# # # # # # #         f"<|system|>Use only the provided Syracuse University research context.<|/system|>\n"
# # # # # # #         f"<|user|>\nQ: {question}\n\nContext:\n" + "\n".join(context) +
# # # # # # #         "\n</|user|>\n<|assistant|>"
# # # # # # #     )


# # # # # # # def answer_question(question: str, n_ctx: int = 6):
# # # # # # #     """Main hybrid retrieval + generation pipeline."""
# # # # # # #     retr = hybrid_search(question, k_chroma=n_ctx, k_graph=n_ctx)
# # # # # # #     ctx = sanitize(retr.get("fused_text_blocks", []))

# # # # # # #     if not ctx:
# # # # # # #         return {
# # # # # # #             "answer": "No relevant information found in Chroma or Graph retrievals.",
# # # # # # #             "sources": [],
# # # # # # #             "graph_hits": []
# # # # # # #         }

# # # # # # #     prompt = build_prompt(question, ctx)
# # # # # # #     tok, model = _load_llm()
# # # # # # #     inputs = tok(prompt, return_tensors="pt").to(model.device)
# # # # # # #     gen = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9)
# # # # # # #     out = tok.decode(gen[0], skip_special_tokens=True)
# # # # # # #     ans = out.split("<|assistant|>")[-1].strip()

# # # # # # #     return {
# # # # # # #         "answer": ans,
# # # # # # #         "sources": ctx,
# # # # # # #         "graph_hits": retr.get("graph_hits", [])
# # # # # # #     }

# # # # # # import re, torch
# # # # # # from transformers import AutoTokenizer, AutoModelForCausalLM
# # # # # # import config_full as config
# # # # # # from hybrid_langchain_retriever import hybrid_search

# # # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # # dtype = torch.float16 if device == "cuda" else torch.float32

# # # # # # tok, model = None, None

# # # # # # def _load_llm():
# # # # # #     global tok, model
# # # # # #     if tok and model:
# # # # # #         return tok, model
# # # # # #     tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
# # # # # #     model = AutoModelForCausalLM.from_pretrained(
# # # # # #         config.LLAMA_MODEL_PATH,
# # # # # #         torch_dtype=dtype,
# # # # # #         device_map="auto" if device == "cuda" else None,
# # # # # #         low_cpu_mem_usage=True,
# # # # # #     )
# # # # # #     if tok.pad_token_id is None and tok.eos_token_id:
# # # # # #         tok.pad_token_id = tok.eos_token_id
# # # # # #     return tok, model


# # # # # # def sanitize(blocks, max_chars=6000):
# # # # # #     out, total = [], 0
# # # # # #     for b in blocks:
# # # # # #         b = re.sub(r"<[^>]+>", " ", b)
# # # # # #         if 20 < len(b) < 500:
# # # # # #             if total + len(b) > max_chars:
# # # # # #                 break
# # # # # #             out.append(b)
# # # # # #             total += len(b)
# # # # # #     return out


# # # # # # def build_prompt(question, context):
# # # # # #     return (
# # # # # #         f"<|system|>Answer using only the provided Syracuse University research context.<|/system|>\n"
# # # # # #         f"<|user|>\nQ: {question}\n\nContext:\n" + "\n".join(context) +
# # # # # #         "\n</|user|>\n<|assistant|>"
# # # # # #     )


# # # # # # def answer_question(question: str, n_ctx: int = 6):
# # # # # #     retr = hybrid_search(question, k_chroma=n_ctx, k_graph=n_ctx)
# # # # # #     ctx = sanitize(retr.get("fused_text_blocks", []))

# # # # # #     if not ctx:
# # # # # #         return {
# # # # # #             "answer": "No relevant information found in Chroma or Graph retrievals.",
# # # # # #             "sources": [],
# # # # # #             "graph_hits": []
# # # # # #         }

# # # # # #     prompt = build_prompt(question, ctx)
# # # # # #     tok, model = _load_llm()
# # # # # #     inputs = tok(prompt, return_tensors="pt").to(model.device)
# # # # # #     gen = model.generate(**inputs, max_new_tokens=256, temperature=0.2, top_p=0.9)
# # # # # #     out = tok.decode(gen[0], skip_special_tokens=True)
# # # # # #     ans = out.split("<|assistant|>")[-1].strip()

# # # # # #     return {
# # # # # #         "answer": ans,
# # # # # #         "sources": ctx,
# # # # # #         "graph_hits": retr.get("graph_hits", [])
# # # # # #     }


# # # # # """
# # # # # rag_pipeline.py - Database-independent RAG pipeline
# # # # # """
# # # # # import re
# # # # # import torch
# # # # # import hashlib
# # # # # from collections import deque
# # # # # from datetime import datetime
# # # # # from transformers import AutoTokenizer, AutoModelForCausalLM
# # # # # import config_full as config
# # # # # from hybrid_langchain_retriever import hybrid_search
# # # # # from database_manager import get_active_db_config

# # # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # # dtype = torch.float16 if device == "cuda" else torch.float32

# # # # # tok, model = None, None

# # # # # # ============================================================================
# # # # # # CONVERSATION MEMORY (unchanged)
# # # # # # ============================================================================
# # # # # class ConversationMemory:
# # # # #     def __init__(self, buffer_size=4, max_summary_tokens=200):
# # # # #         self.buffer_size = buffer_size
# # # # #         self.max_summary_tokens = max_summary_tokens
# # # # #         self.buffer = deque(maxlen=buffer_size)
# # # # #         self.summary = ""
# # # # #         self.archived_count = 0
    
# # # # #     def add_message(self, role: str, content: str):
# # # # #         self.buffer.append({
# # # # #             "role": role,
# # # # #             "content": content,
# # # # #             "timestamp": datetime.now().isoformat()
# # # # #         })
        
# # # # #         if len(self.buffer) >= self.buffer_size:
# # # # #             self._maybe_summarize()
    
# # # # #     def _maybe_summarize(self):
# # # # #         if len(self.buffer) < self.buffer_size or self.archived_count >= 20:
# # # # #             return
        
# # # # #         to_summarize = [self.buffer.popleft() for _ in range(2) if self.buffer]
        
# # # # #         if to_summarize:
# # # # #             summary_text = "\n".join([
# # # # #                 f"{msg['role']}: {msg['content'][:100]}..." 
# # # # #                 for msg in to_summarize
# # # # #             ])
            
# # # # #             if self.summary:
# # # # #                 self.summary += f"\n[Earlier: {summary_text}]"
# # # # #             else:
# # # # #                 self.summary = f"[Earlier conversation: {summary_text}]"
            
# # # # #             self.archived_count += len(to_summarize)
    
# # # # #     def get_context(self, max_chars=800) -> str:
# # # # #         context_parts = []
        
# # # # #         if self.summary:
# # # # #             context_parts.append(f"Summary of earlier conversation:\n{self.summary[:max_chars//2]}")
        
# # # # #         for msg in self.buffer:
# # # # #             role_label = "User" if msg["role"] == "user" else "Assistant"
# # # # #             context_parts.append(f"{role_label}: {msg['content'][:max_chars//len(self.buffer)]}")
        
# # # # #         return "\n\n".join(context_parts)
    
# # # # #     def clear(self):
# # # # #         self.buffer.clear()
# # # # #         self.summary = ""
# # # # #         self.archived_count = 0


# # # # # # ============================================================================
# # # # # # RETRIEVAL CACHING (unchanged)
# # # # # # ============================================================================
# # # # # class RetrievalCache:
# # # # #     def __init__(self, maxsize=100):
# # # # #         self.cache = {}
# # # # #         self.access_order = deque(maxlen=maxsize)
# # # # #         self.maxsize = maxsize
# # # # #         self.hits = 0
# # # # #         self.misses = 0
    
# # # # #     def _make_key(self, query: str, db_config_name: str, n_ctx: int) -> str:
# # # # #         key_string = f"{query.lower().strip()}_{db_config_name}_{n_ctx}"
# # # # #         return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
# # # # #     def get(self, query: str, db_config_name: str, n_ctx: int):
# # # # #         key = self._make_key(query, db_config_name, n_ctx)
        
# # # # #         if key in self.cache:
# # # # #             self.hits += 1
# # # # #             self.access_order.remove(key)
# # # # #             self.access_order.append(key)
# # # # #             return self.cache[key]
        
# # # # #         self.misses += 1
# # # # #         return None
    
# # # # #     def set(self, query: str, db_config_name: str, n_ctx: int, result):
# # # # #         key = self._make_key(query, db_config_name, n_ctx)
        
# # # # #         if len(self.cache) >= self.maxsize and key not in self.cache:
# # # # #             lru_key = self.access_order.popleft()
# # # # #             del self.cache[lru_key]
        
# # # # #         self.cache[key] = result
# # # # #         if key in self.access_order:
# # # # #             self.access_order.remove(key)
# # # # #         self.access_order.append(key)
    
# # # # #     def clear(self):
# # # # #         self.cache.clear()
# # # # #         self.access_order.clear()
# # # # #         self.hits = 0
# # # # #         self.misses = 0
    
# # # # #     def stats(self):
# # # # #         total = self.hits + self.misses
# # # # #         hit_rate = (self.hits / total * 100) if total > 0 else 0
# # # # #         return {
# # # # #             "size": len(self.cache),
# # # # #             "maxsize": self.maxsize,
# # # # #             "hits": self.hits,
# # # # #             "misses": self.misses,
# # # # #             "hit_rate": f"{hit_rate:.1f}%"
# # # # #         }


# # # # # # Global instances
# # # # # retrieval_cache = RetrievalCache(maxsize=100)
# # # # # conversation_memory = ConversationMemory(buffer_size=4)


# # # # # # ============================================================================
# # # # # # MODEL LOADING (unchanged)
# # # # # # ============================================================================
# # # # # def _load_llm():
# # # # #     global tok, model
# # # # #     if tok and model:
# # # # #         return tok, model
    
# # # # #     print("Loading LLaMA model...")
# # # # #     tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
# # # # #     model = AutoModelForCausalLM.from_pretrained(
# # # # #         config.LLAMA_MODEL_PATH,
# # # # #         torch_dtype=dtype,
# # # # #         device_map="auto" if device == "cuda" else None,
# # # # #         low_cpu_mem_usage=True,
# # # # #     )
    
# # # # #     if tok.pad_token_id is None and tok.eos_token_id:
# # # # #         tok.pad_token_id = tok.eos_token_id
    
# # # # #     print(f"✅ Model loaded on {device}")
# # # # #     return tok, model


# # # # # # ============================================================================
# # # # # # TEXT PROCESSING
# # # # # # ============================================================================
# # # # # def sanitize(blocks, max_chars=5000):
# # # # #     out, total = [], 0
    
# # # # #     for b in blocks:
# # # # #         b = re.sub(r"<[^>]+>", " ", str(b))
# # # # #         b = re.sub(r"\s+", " ", b).strip()
        
# # # # #         if 30 < len(b) < 600:
# # # # #             if total + len(b) > max_chars:
# # # # #                 break
# # # # #             out.append(b)
# # # # #             total += len(b)
    
# # # # #     return out


# # # # # def build_prompt(question, context, db_config, conversation_context=""):
# # # # #     """Build prompt with database description."""
# # # # #     context_text = "\n".join([f"{i+1}. {ctx}" for i, ctx in enumerate(context)])
    
# # # # #     prompt = f"<|system|>You are a research assistant for Syracuse University. "
# # # # #     prompt += f"Answer using ONLY the provided context from {db_config.description}. "
# # # # #     prompt += f"Context is ordered chronologically (most recent first). "
# # # # #     prompt += f"Prioritize recent papers when relevant."
    
# # # # #     if conversation_context:
# # # # #         prompt += f"\n\nConversation history:\n{conversation_context}"
    
# # # # #     prompt += f"<|/system|>\n<|user|>\nQuestion: {question}\n\n"
# # # # #     prompt += f"Context:\n{context_text}\n</|user|>\n<|assistant|>"
    
# # # # #     return prompt


# # # # # # ============================================================================
# # # # # # MAIN RAG PIPELINE (Database-independent)
# # # # # # ============================================================================
# # # # # def answer_question(
# # # # #     question: str, 
# # # # #     n_ctx: int = 6,
# # # # #     use_cache: bool = True,
# # # # #     use_conversation: bool = True
# # # # # ):
# # # # #     """
# # # # #     Database-independent RAG pipeline.
# # # # #     Automatically uses active database configuration.
# # # # #     """
    
# # # # #     # Get active database config
# # # # #     db_config = get_active_db_config()
# # # # #     config_name = get_db_manager().active_config_name
    
# # # # #     # Check cache
# # # # #     cache_hit = False
# # # # #     if use_cache:
# # # # #         cached = retrieval_cache.get(question, config_name, n_ctx)
# # # # #         if cached:
# # # # #             print("⚡ Cache hit!")
# # # # #             cache_hit = True
# # # # #             retr = cached
    
# # # # #     # Retrieve if not cached
# # # # #     if not cache_hit:
# # # # #         try:
# # # # #             retr = hybrid_search(question, k_chroma=n_ctx, k_graph=n_ctx)
            
# # # # #             if use_cache:
# # # # #                 retrieval_cache.set(question, config_name, n_ctx, retr)
                
# # # # #         except Exception as e:
# # # # #             print(f"❌ Retrieval error: {e}")
# # # # #             return {
# # # # #                 "answer": f"Error retrieving information from {db_config.description}.",
# # # # #                 "sources": [],
# # # # #                 "graph_hits": [],
# # # # #                 "cache_hit": False,
# # # # #                 "conversation_used": False,
# # # # #                 "db_config": config_name
# # # # #             }
    
# # # # #     # Sanitize context
# # # # #     ctx = sanitize(retr.get("fused_text_blocks", []))
    
# # # # #     if not ctx:
# # # # #         return {
# # # # #             "answer": f"No relevant information found in {db_config.description}.",
# # # # #             "sources": [],
# # # # #             "graph_hits": [],
# # # # #             "cache_hit": cache_hit,
# # # # #             "conversation_used": False,
# # # # #             "db_config": config_name
# # # # #         }
    
# # # # #     # Get conversation context
# # # # #     conversation_context = ""
# # # # #     if use_conversation:
# # # # #         conversation_context = conversation_memory.get_context()
    
# # # # #     # Generate answer
# # # # #     try:
# # # # #         prompt = build_prompt(question, ctx, db_config, conversation_context)
# # # # #         tok, model = _load_llm()
        
# # # # #         inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
# # # # #         inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
# # # # #         gen = model.generate(
# # # # #             **inputs,
# # # # #             max_new_tokens=256,
# # # # #             temperature=0.2,
# # # # #             top_p=0.9,
# # # # #             do_sample=True,
# # # # #             pad_token_id=tok.pad_token_id
# # # # #         )
        
# # # # #         out = tok.decode(gen[0], skip_special_tokens=True)
# # # # #         ans = out.split("<|assistant|>")[-1].strip()
        
# # # # #         if not ans or len(ans) < 10:
# # # # #             ans = "Unable to generate a complete answer. Please rephrase your question."
    
# # # # #     except Exception as e:
# # # # #         print(f"❌ Generation error: {e}")
# # # # #         ans = "Error generating answer from the model."
    
# # # # #     # Add to conversation memory
# # # # #     if use_conversation:
# # # # #         conversation_memory.add_message("user", question)
# # # # #         conversation_memory.add_message("assistant", ans)
    
# # # # #     return {
# # # # #         "answer": ans,
# # # # #         "sources": ctx,
# # # # #         "graph_hits": retr.get("graph_hits", []),
# # # # #         "cache_hit": cache_hit,
# # # # #         "conversation_used": use_conversation,
# # # # #         "db_config": config_name
# # # # #     }


# # # # # def clear_cache():
# # # # #     retrieval_cache.clear()
# # # # #     print("🗑️ Cache cleared")


# # # # # def clear_conversation():
# # # # #     conversation_memory.clear()
# # # # #     print("🗑️ Conversation memory cleared")


# # # # # def get_cache_stats():
# # # # #     return retrieval_cache.stats()


# # # # # def get_conversation_summary():
# # # # #     return {
# # # # #         "buffer_size": len(conversation_memory.buffer),
# # # # #         "archived_count": conversation_memory.archived_count,
# # # # #         "has_summary": bool(conversation_memory.summary)
# # # # #     }


# # # # # # Pre-warm model
# # # # # try:
# # # # #     _load_llm()
# # # # # except Exception as e:
# # # # #     print(f"⚠️ Model pre-warming skipped: {e}")


# # # # # # Import at module level for convenience
# # # # # from database_manager import get_db_manager

# # # # """
# # # # rag_pipeline.py - Updated with better prompt
# # # # """
# # # # """
# # # # rag_pipeline.py - Fixed answer generation
# # # # """
# # # # import re
# # # # import torch
# # # # import hashlib
# # # # from collections import deque
# # # # from datetime import datetime
# # # # from transformers import AutoTokenizer, AutoModelForCausalLM
# # # # import config_full as config
# # # # from hybrid_langchain_retriever import hybrid_search
# # # # from database_manager import get_active_db_config, get_db_manager

# # # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # dtype = torch.float16 if device == "cuda" else torch.float32

# # # # tok, model = None, None

# # # # # ============================================================================
# # # # # CONVERSATION MEMORY
# # # # # ============================================================================
# # # # class ConversationMemory:
# # # #     def __init__(self, buffer_size=4, max_summary_tokens=200):
# # # #         self.buffer_size = buffer_size
# # # #         self.max_summary_tokens = max_summary_tokens
# # # #         self.buffer = deque(maxlen=buffer_size)
# # # #         self.summary = ""
# # # #         self.archived_count = 0
    
# # # #     def add_message(self, role: str, content: str):
# # # #         self.buffer.append({
# # # #             "role": role,
# # # #             "content": content,
# # # #             "timestamp": datetime.now().isoformat()
# # # #         })
        
# # # #         if len(self.buffer) >= self.buffer_size:
# # # #             self._maybe_summarize()
    
# # # #     def _maybe_summarize(self):
# # # #         if len(self.buffer) < self.buffer_size or self.archived_count >= 20:
# # # #             return
        
# # # #         to_summarize = [self.buffer.popleft() for _ in range(2) if self.buffer]
        
# # # #         if to_summarize:
# # # #             summary_text = "\n".join([
# # # #                 f"{msg['role']}: {msg['content'][:100]}..." 
# # # #                 for msg in to_summarize
# # # #             ])
            
# # # #             if self.summary:
# # # #                 self.summary += f"\n[Earlier: {summary_text}]"
# # # #             else:
# # # #                 self.summary = f"[Earlier conversation: {summary_text}]"
            
# # # #             self.archived_count += len(to_summarize)
    
# # # #     def get_context(self, max_chars=600) -> str:
# # # #         context_parts = []
        
# # # #         if self.summary:
# # # #             context_parts.append(f"Summary: {self.summary[:max_chars//2]}")
        
# # # #         for msg in self.buffer:
# # # #             role_label = "User" if msg["role"] == "user" else "Assistant"
# # # #             context_parts.append(f"{role_label}: {msg['content'][:max_chars//len(self.buffer)]}")
        
# # # #         return "\n".join(context_parts)
    
# # # #     def clear(self):
# # # #         self.buffer.clear()
# # # #         self.summary = ""
# # # #         self.archived_count = 0


# # # # # ============================================================================
# # # # # RETRIEVAL CACHING
# # # # # ============================================================================
# # # # class RetrievalCache:
# # # #     def __init__(self, maxsize=100):
# # # #         self.cache = {}
# # # #         self.access_order = deque(maxlen=maxsize)
# # # #         self.maxsize = maxsize
# # # #         self.hits = 0
# # # #         self.misses = 0
    
# # # #     def _make_key(self, query: str, db_config_name: str, n_ctx: int) -> str:
# # # #         key_string = f"{query.lower().strip()}_{db_config_name}_{n_ctx}"
# # # #         return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
# # # #     def get(self, query: str, db_config_name: str, n_ctx: int):
# # # #         key = self._make_key(query, db_config_name, n_ctx)
        
# # # #         if key in self.cache:
# # # #             self.hits += 1
# # # #             self.access_order.remove(key)
# # # #             self.access_order.append(key)
# # # #             return self.cache[key]
        
# # # #         self.misses += 1
# # # #         return None
    
# # # #     def set(self, query: str, db_config_name: str, n_ctx: int, result):
# # # #         key = self._make_key(query, db_config_name, n_ctx)
        
# # # #         if len(self.cache) >= self.maxsize and key not in self.cache:
# # # #             lru_key = self.access_order.popleft()
# # # #             del self.cache[lru_key]
        
# # # #         self.cache[key] = result
# # # #         if key in self.access_order:
# # # #             self.access_order.remove(key)
# # # #         self.access_order.append(key)
    
# # # #     def clear(self):
# # # #         self.cache.clear()
# # # #         self.access_order.clear()
# # # #         self.hits = 0
# # # #         self.misses = 0
    
# # # #     def stats(self):
# # # #         total = self.hits + self.misses
# # # #         hit_rate = (self.hits / total * 100) if total > 0 else 0
# # # #         return {
# # # #             "size": len(self.cache),
# # # #             "maxsize": self.maxsize,
# # # #             "hits": self.hits,
# # # #             "misses": self.misses,
# # # #             "hit_rate": f"{hit_rate:.1f}%"
# # # #         }


# # # # # Global instances
# # # # retrieval_cache = RetrievalCache(maxsize=100)
# # # # conversation_memory = ConversationMemory(buffer_size=4)


# # # # # ============================================================================
# # # # # MODEL LOADING
# # # # # ============================================================================
# # # # def _load_llm():
# # # #     global tok, model
# # # #     if tok and model:
# # # #         return tok, model
    
# # # #     print("Loading LLaMA model...")
# # # #     tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
# # # #     model = AutoModelForCausalLM.from_pretrained(
# # # #         config.LLAMA_MODEL_PATH,
# # # #         torch_dtype=dtype,
# # # #         device_map="auto" if device == "cuda" else None,
# # # #         low_cpu_mem_usage=True,
# # # #     )
    
# # # #     if tok.pad_token_id is None and tok.eos_token_id:
# # # #         tok.pad_token_id = tok.eos_token_id
    
# # # #     print(f"✅ Model loaded on {device}")
# # # #     return tok, model


# # # # # ============================================================================
# # # # # TEXT PROCESSING
# # # # # ============================================================================
# # # # def sanitize(blocks, max_chars=4000):
# # # #     """Clean and filter context blocks."""
# # # #     out, total = [], 0
    
# # # #     for b in blocks:
# # # #         b = re.sub(r"<[^>]+>", " ", str(b))
# # # #         b = re.sub(r"\s+", " ", b).strip()
        
# # # #         if 30 < len(b) < 500:
# # # #             if total + len(b) > max_chars:
# # # #                 break
# # # #             out.append(b)
# # # #             total += len(b)
    
# # # #     return out


# # # # def build_prompt(question, context, conversation_context=""):
# # # #     """
# # # #     Build prompt in the format that works best.
# # # #     """
# # # #     context_text = "\n".join(context)
    
# # # #     system_msg = "Answer using only the provided Syracuse University research context."
    
# # # #     if conversation_context:
# # # #         system_msg += f"\n\nPrevious conversation:\n{conversation_context}"
    
# # # #     prompt = (
# # # #         f"<|system|>{system_msg}<|/system|>\n"
# # # #         f"<|user|>\nQ: {question}\n\nContext:\n{context_text}\n<|/user|>\n"
# # # #         f"<|assistant|>"
# # # #     )
    
# # # #     return prompt


# # # # # ============================================================================
# # # # # MAIN RAG PIPELINE
# # # # # ============================================================================
# # # # def answer_question(
# # # #     question: str, 
# # # #     n_ctx: int = 6,
# # # #     use_cache: bool = True,
# # # #     use_conversation: bool = True
# # # # ):
# # # #     """RAG pipeline with better answer generation."""
    
# # # #     db_config = get_active_db_config()
# # # #     config_name = get_db_manager().active_config_name
    
# # # #     # Check cache
# # # #     cache_hit = False
# # # #     if use_cache:
# # # #         cached = retrieval_cache.get(question, config_name, n_ctx)
# # # #         if cached:
# # # #             print("⚡ Cache hit!")
# # # #             cache_hit = True
# # # #             retr = cached
    
# # # #     # Retrieve if not cached
# # # #     if not cache_hit:
# # # #         try:
# # # #             retr = hybrid_search(question, k_chroma=n_ctx, k_graph=n_ctx)
            
# # # #             if use_cache:
# # # #                 retrieval_cache.set(question, config_name, n_ctx, retr)
                
# # # #         except Exception as e:
# # # #             print(f"❌ Retrieval error: {e}")
# # # #             return {
# # # #                 "answer": f"Error retrieving information from {db_config.description}.",
# # # #                 "sources": [],
# # # #                 "graph_hits": [],
# # # #                 "cache_hit": False,
# # # #                 "conversation_used": False,
# # # #                 "db_config": config_name
# # # #             }
    
# # # #     # Sanitize context
# # # #     ctx = sanitize(retr.get("fused_text_blocks", []))
    
# # # #     if not ctx:
# # # #         return {
# # # #             "answer": f"No relevant information found in {db_config.description}.",
# # # #             "sources": [],
# # # #             "graph_hits": [],
# # # #             "cache_hit": cache_hit,
# # # #             "conversation_used": False,
# # # #             "db_config": config_name
# # # #         }
    
# # # #     # Get conversation context (shorter for better generation)
# # # #     conversation_context = ""
# # # #     if use_conversation:
# # # #         conversation_context = conversation_memory.get_context(max_chars=400)
    
# # # #     # Generate answer
# # # #     try:
# # # #         prompt = build_prompt(question, ctx, conversation_context)
# # # #         tok, model = _load_llm()
        
# # # #         inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1800)
# # # #         inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
# # # #         # FIXED: Increase max_new_tokens and adjust parameters
# # # #         gen = model.generate(
# # # #             **inputs,
# # # #             max_new_tokens=400,  # Increased from 256
# # # #             temperature=0.3,     # Slightly higher for more variation
# # # #             top_p=0.92,
# # # #             do_sample=True,
# # # #             repetition_penalty=1.1,  # Prevent repetition
# # # #             pad_token_id=tok.pad_token_id,
# # # #             eos_token_id=tok.eos_token_id
# # # #         )
        
# # # #         out = tok.decode(gen[0], skip_special_tokens=True)
        
# # # #         # Extract answer (everything after <|assistant|>)
# # # #         if "<|assistant|>" in out:
# # # #             ans = out.split("<|assistant|>")[-1].strip()
# # # #         else:
# # # #             ans = out.strip()
        
# # # #         # Remove any trailing incomplete sentences
# # # #         if ans and not ans[-1] in '.!?':
# # # #             # Find last complete sentence
# # # #             last_period = max(ans.rfind('.'), ans.rfind('!'), ans.rfind('?'))
# # # #             if last_period > 50:  # Keep if there's a reasonable amount of text
# # # #                 ans = ans[:last_period + 1]
        
# # # #         if not ans or len(ans) < 20:
# # # #             ans = "Unable to generate a complete answer from the retrieved context. Please try rephrasing your question."
    
# # # #     except Exception as e:
# # # #         print(f"❌ Generation error: {e}")
# # # #         import traceback
# # # #         traceback.print_exc()
# # # #         ans = f"Error generating answer: {str(e)}"
    
# # # #     # Add to conversation memory
# # # #     if use_conversation:
# # # #         conversation_memory.add_message("user", question)
# # # #         conversation_memory.add_message("assistant", ans)
    
# # # #     return {
# # # #         "answer": ans,
# # # #         "sources": ctx,
# # # #         "graph_hits": retr.get("graph_hits", []),
# # # #         "cache_hit": cache_hit,
# # # #         "conversation_used": use_conversation,
# # # #         "db_config": config_name
# # # #     }


# # # # def clear_cache():
# # # #     retrieval_cache.clear()
# # # #     print("🗑️ Cache cleared")


# # # # def clear_conversation():
# # # #     conversation_memory.clear()
# # # #     print("🗑️ Conversation memory cleared")


# # # # def get_cache_stats():
# # # #     return retrieval_cache.stats()


# # # # def get_conversation_summary():
# # # #     return {
# # # #         "buffer_size": len(conversation_memory.buffer),
# # # #         "archived_count": conversation_memory.archived_count,
# # # #         "has_summary": bool(conversation_memory.summary)
# # # #     }


# # # # # Pre-warm model
# # # # try:
# # # #     _load_llm()
# # # # except Exception as e:
# # # #     print(f"⚠️ Model pre-warming skipped: {e}")

# # # """
# # # rag_pipeline.py - Pure semantic RAG pipeline
# # # """
# # # import re
# # # import torch
# # # import hashlib
# # # from collections import deque
# # # from datetime import datetime
# # # from transformers import AutoTokenizer, AutoModelForCausalLM
# # # import config_full as config
# # # from hybrid_langchain_retriever import hybrid_search
# # # from database_manager import get_active_db_config, get_db_manager

# # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # dtype = torch.float16 if device == "cuda" else torch.float32

# # # tok, model = None, None


# # # # ============================================================================
# # # # CONVERSATION MEMORY
# # # # ============================================================================
# # # class ConversationMemory:
# # #     def __init__(self, buffer_size=4):
# # #         self.buffer_size = buffer_size
# # #         self.buffer = deque(maxlen=buffer_size)
# # #         self.summary = ""
# # #         self.archived_count = 0
    
# # #     def add_message(self, role: str, content: str):
# # #         self.buffer.append({
# # #             "role": role,
# # #             "content": content,
# # #             "timestamp": datetime.now().isoformat()
# # #         })
        
# # #         if len(self.buffer) >= self.buffer_size:
# # #             self._maybe_summarize()
    
# # #     def _maybe_summarize(self):
# # #         if len(self.buffer) < self.buffer_size or self.archived_count >= 20:
# # #             return
        
# # #         to_summarize = [self.buffer.popleft() for _ in range(2) if self.buffer]
        
# # #         if to_summarize:
# # #             summary_text = "\n".join([
# # #                 f"{msg['role']}: {msg['content'][:100]}..." 
# # #                 for msg in to_summarize
# # #             ])
            
# # #             if self.summary:
# # #                 self.summary += f"\n[Earlier: {summary_text}]"
# # #             else:
# # #                 self.summary = f"[Earlier: {summary_text}]"
            
# # #             self.archived_count += len(to_summarize)
    
# # #     def get_context(self, max_chars=600) -> str:
# # #         context_parts = []
        
# # #         if self.summary:
# # #             context_parts.append(self.summary[:max_chars//2])
        
# # #         for msg in list(self.buffer)[-3:]:  # Last 3 messages only
# # #             role = "User" if msg["role"] == "user" else "Assistant"
# # #             context_parts.append(f"{role}: {msg['content'][:200]}")
        
# # #         return "\n".join(context_parts)
    
# # #     def clear(self):
# # #         self.buffer.clear()
# # #         self.summary = ""
# # #         self.archived_count = 0


# # # # ============================================================================
# # # # RETRIEVAL CACHING
# # # # ============================================================================
# # # class RetrievalCache:
# # #     def __init__(self, maxsize=100):
# # #         self.cache = {}
# # #         self.access_order = deque(maxlen=maxsize)
# # #         self.maxsize = maxsize
# # #         self.hits = 0
# # #         self.misses = 0
    
# # #     def _make_key(self, query: str, db_config_name: str, n_ctx: int) -> str:
# # #         key_string = f"{query.lower().strip()}_{db_config_name}_{n_ctx}"
# # #         return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
# # #     def get(self, query: str, db_config_name: str, n_ctx: int):
# # #         key = self._make_key(query, db_config_name, n_ctx)
        
# # #         if key in self.cache:
# # #             self.hits += 1
# # #             self.access_order.remove(key)
# # #             self.access_order.append(key)
# # #             return self.cache[key]
        
# # #         self.misses += 1
# # #         return None
    
# # #     def set(self, query: str, db_config_name: str, n_ctx: int, result):
# # #         key = self._make_key(query, db_config_name, n_ctx)
        
# # #         if len(self.cache) >= self.maxsize and key not in self.cache:
# # #             lru_key = self.access_order.popleft()
# # #             del self.cache[lru_key]
        
# # #         self.cache[key] = result
# # #         if key in self.access_order:
# # #             self.access_order.remove(key)
# # #         self.access_order.append(key)
    
# # #     def clear(self):
# # #         self.cache.clear()
# # #         self.access_order.clear()
# # #         self.hits = 0
# # #         self.misses = 0
    
# # #     def stats(self):
# # #         total = self.hits + self.misses
# # #         hit_rate = (self.hits / total * 100) if total > 0 else 0
# # #         return {
# # #             "size": len(self.cache),
# # #             "maxsize": self.maxsize,
# # #             "hits": self.hits,
# # #             "misses": self.misses,
# # #             "hit_rate": f"{hit_rate:.1f}%"
# # #         }


# # # retrieval_cache = RetrievalCache(maxsize=100)
# # # conversation_memory = ConversationMemory(buffer_size=4)


# # # # ============================================================================
# # # # MODEL LOADING
# # # # ============================================================================
# # # def _load_llm():
# # #     global tok, model
# # #     if tok and model:
# # #         return tok, model
    
# # #     print("Loading LLaMA model...")
# # #     tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH)
# # #     model = AutoModelForCausalLM.from_pretrained(
# # #         config.LLAMA_MODEL_PATH,
# # #         dtype=torch.float16 if device == "cuda" else torch.float32,
# # #         device_map="auto" if device == "cuda" else None,
# # #         low_cpu_mem_usage=True,
# # #     )
    
# # #     if tok.pad_token_id is None and tok.eos_token_id:
# # #         tok.pad_token_id = tok.eos_token_id
    
# # #     print(f"✅ Model loaded on {device}")
# # #     return tok, model


# # # # ============================================================================
# # # # TEXT PROCESSING
# # # # ============================================================================
# # # def sanitize(blocks, max_chars=4000):
# # #     out, total = [], 0
    
# # #     for b in blocks:
# # #         b = re.sub(r"<[^>]+>", " ", str(b))
# # #         b = re.sub(r"\s+", " ", b).strip()
        
# # #         if 30 < len(b) < 500:
# # #             if total + len(b) > max_chars:
# # #                 break
# # #             out.append(b)
# # #             total += len(b)
    
# # #     return out


# # # def build_prompt(question, context, conversation_context=""):
# # #     """Simple effective prompt."""
# # #     context_text = "\n".join(context)
    
# # #     system_msg = "Answer using only the provided Syracuse University research context."
    
# # #     if conversation_context:
# # #         system_msg += f"\n\nPrevious conversation:\n{conversation_context}"
    
# # #     prompt = (
# # #         f"<|system|>{system_msg}<|/system|>\n"
# # #         f"<|user|>\nQ: {question}\n\nContext:\n{context_text}\n<|/user|>\n"
# # #         f"<|assistant|>"
# # #     )
    
# # #     return prompt


# # # # ============================================================================
# # # # MAIN RAG PIPELINE
# # # # ============================================================================
# # # def answer_question(
# # #     question: str, 
# # #     n_ctx: int = 6,
# # #     use_cache: bool = True,
# # #     use_conversation: bool = True
# # # ):
# # #     """Pure semantic RAG pipeline."""
    
# # #     db_config = get_active_db_config()
# # #     config_name = get_db_manager().active_config_name
    
# # #     # Check cache
# # #     cache_hit = False
# # #     if use_cache:
# # #         cached = retrieval_cache.get(question, config_name, n_ctx)
# # #         if cached:
# # #             print("⚡ Cache hit!")
# # #             cache_hit = True
# # #             retr = cached
    
# # #     # Retrieve if not cached
# # #     if not cache_hit:
# # #         try:
# # #             retr = hybrid_search(question, k_chroma=n_ctx, k_graph=n_ctx)
            
# # #             if use_cache:
# # #                 retrieval_cache.set(question, config_name, n_ctx, retr)
                
# # #         except Exception as e:
# # #             print(f"❌ Retrieval error: {e}")
# # #             return {
# # #                 "answer": f"Error: {str(e)}",
# # #                 "sources": [],
# # #                 "graph_hits": [],
# # #                 "cache_hit": False,
# # #                 "conversation_used": False,
# # #                 "db_config": config_name
# # #             }
    
# # #     ctx = sanitize(retr.get("fused_text_blocks", []))
    
# # #     if not ctx:
# # #         return {
# # #             "answer": "No relevant papers found.",
# # #             "sources": [],
# # #             "graph_hits": [],
# # #             "cache_hit": cache_hit,
# # #             "conversation_used": False,
# # #             "db_config": config_name
# # #         }
    
# # #     # Get conversation context
# # #     conversation_context = ""
# # #     if use_conversation:
# # #         conversation_context = conversation_memory.get_context()
    
# # #     # Generate answer
# # #     try:
# # #         prompt = build_prompt(question, ctx, conversation_context)
# # #         tok, model = _load_llm()
        
# # #         inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1800)
# # #         inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
# # #         gen = model.generate(
# # #             **inputs,
# # #             max_new_tokens=400,
# # #             temperature=0.3,
# # #             top_p=0.92,
# # #             do_sample=True,
# # #             repetition_penalty=1.1,
# # #             pad_token_id=tok.pad_token_id,
# # #             eos_token_id=tok.eos_token_id
# # #         )
        
# # #         out = tok.decode(gen[0], skip_special_tokens=True)
        
# # #         if "<|assistant|>" in out:
# # #             ans = out.split("<|assistant|>")[-1].strip()
# # #         else:
# # #             ans = out.strip()
        
# # #         # Clean up incomplete sentences
# # #         if ans and ans[-1] not in '.!?':
# # #             last_period = max(ans.rfind('.'), ans.rfind('!'), ans.rfind('?'))
# # #             if last_period > 50:
# # #                 ans = ans[:last_period + 1]
        
# # #         if not ans or len(ans) < 20:
# # #             ans = "Unable to generate answer. Please rephrase your question."
    
# # #     except Exception as e:
# # #         print(f"❌ Generation error: {e}")
# # #         ans = f"Error generating answer: {str(e)}"
    
# # #     if use_conversation:
# # #         conversation_memory.add_message("user", question)
# # #         conversation_memory.add_message("assistant", ans)
    
# # #     return {
# # #         "answer": ans,
# # #         "sources": ctx,
# # #         "graph_hits": retr.get("graph_hits", []),
# # #         "cache_hit": cache_hit,
# # #         "conversation_used": use_conversation,
# # #         "db_config": config_name
# # #     }


# # # def clear_cache():
# # #     retrieval_cache.clear()


# # # def clear_conversation():
# # #     conversation_memory.clear()


# # # def get_cache_stats():
# # #     return retrieval_cache.stats()


# # # def get_conversation_summary():
# # #     return {
# # #         "buffer_size": len(conversation_memory.buffer),
# # #         "archived_count": conversation_memory.archived_count,
# # #         "has_summary": bool(conversation_memory.summary)
# # #     }


# # # try:
# # #     _load_llm()
# # # except Exception as e:
# # #     print(f"⚠️ Model pre-warming skipped: {e}")

# # """
# # rag_pipeline.py — SyracuseRAG-LLaMA Final Stable Version (Nov 2025)
# # Hybrid RAG with deterministic LLaMA-3.2-1B generation
# # ------------------------------------------------------
# # ✓ Correct answers when context is semantically matched
# # ✓ Prompt compatible with LLaMA-3.2-1B-Instruct (no tag confusion)
# # ✓ Context length increased (no truncated abstracts)
# # ✓ Cache stats restored for Streamlit display
# # ✓ Deterministic + reproducible generation
# # """

# # import re, torch, hashlib
# # from collections import deque
# # from datetime import datetime
# # from transformers import AutoTokenizer, AutoModelForCausalLM

# # import config_full as config
# # from hybrid_langchain_retriever import hybrid_search
# # from database_manager import get_active_db_config, get_db_manager


# # # ==============================================================
# # # DEVICE + MODEL SETTINGS
# # # ==============================================================
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # dtype  = torch.float16 if device == "cuda" else torch.float32
# # tok, model = None, None


# # def _load_llm():
# #     """Lazy-load LLaMA only once."""
# #     global tok, model
# #     if tok and model:
# #         return tok, model
# #     print(f"⚙️  Loading LLaMA model from {config.LLAMA_MODEL_PATH} …")
# #     tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH, use_fast=True)
# #     model = AutoModelForCausalLM.from_pretrained(
# #         config.LLAMA_MODEL_PATH,
# #         dtype=dtype,
# #         device_map="auto" if device == "cuda" else None,
# #         low_cpu_mem_usage=True,
# #     )
# #     if tok.pad_token_id is None and tok.eos_token_id:
# #         tok.pad_token_id = tok.eos_token_id
# #     print(f"✅ Model loaded on {device}")
# #     return tok, model


# # # ==============================================================
# # # CONVERSATION MEMORY
# # # ==============================================================
# # class ConversationMemory:
# #     def __init__(self, buffer_size=4):
# #         self.buffer = deque(maxlen=buffer_size)
# #         self.summary = ""
# #         self.archived = 0

# #     def add_message(self, role, content):
# #         self.buffer.append(
# #             {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
# #         )
# #         if len(self.buffer) >= self.buffer.maxlen:
# #             self._summarize_partial()

# #     def _summarize_partial(self):
# #         if self.archived >= 20 or len(self.buffer) < 2:
# #             return
# #         msgs = [self.buffer.popleft() for _ in range(2)]
# #         snippet = "\n".join([f"{m['role']}: {m['content'][:120]}…" for m in msgs])
# #         self.summary = f"{self.summary}\n[Earlier] {snippet}" if self.summary else snippet
# #         self.archived += 2

# #     def get_context(self, max_chars=600):
# #         parts = []
# #         if self.summary:
# #             parts.append(self.summary[: max_chars // 2])
# #         for m in list(self.buffer)[-3:]:
# #             parts.append(f"{m['role'].capitalize()}: {m['content'][:200]}")
# #         return "\n".join(parts)

# #     def clear(self):
# #         self.buffer.clear()
# #         self.summary = ""
# #         self.archived = 0


# # # ==============================================================
# # # RETRIEVAL CACHE
# # # ==============================================================
# # class RetrievalCache:
# #     def __init__(self, maxsize=100):
# #         self.cache, self.order = {}, deque(maxlen=maxsize)
# #         self.maxsize = maxsize
# #         self.hits = self.misses = 0

# #     def _key(self, q, db, n):
# #         return hashlib.sha256(f"{q.lower()}_{db}_{n}".encode()).hexdigest()[:16]

# #     def get(self, q, db, n):
# #         k = self._key(q, db, n)
# #         if k in self.cache:
# #             self.hits += 1
# #             self.order.remove(k)
# #             self.order.append(k)
# #             return self.cache[k]
# #         self.misses += 1
# #         return None

# #     def set(self, q, db, n, val):
# #         k = self._key(q, db, n)
# #         if len(self.cache) >= self.maxsize and k not in self.cache:
# #             old = self.order.popleft()
# #             del self.cache[old]
# #         self.cache[k] = val
# #         if k in self.order:
# #             self.order.remove(k)
# #         self.order.append(k)

# #     def stats(self):
# #         """Restored full stats for Streamlit compatibility."""
# #         total = self.hits + self.misses
# #         hit_rate = f"{(self.hits / total * 100):.1f}%" if total else "0%"
# #         return {
# #             "size": len(self.cache),
# #             "maxsize": self.maxsize,
# #             "hits": self.hits,
# #             "misses": self.misses,
# #             "hit_rate": hit_rate,
# #         }

# #     def clear(self):
# #         self.cache.clear()
# #         self.order.clear()
# #         self.hits = self.misses = 0


# # retrieval_cache = RetrievalCache()
# # conversation_memory = ConversationMemory()


# # # ==============================================================
# # # SANITIZATION
# # # ==============================================================
# # def sanitize(blocks, max_chars=6000):
# #     """Strip HTML and preserve larger abstract chunks."""
# #     out, total = [], 0
# #     for b in blocks:
# #         b = re.sub(r"<[^>]+>", " ", str(b))
# #         b = re.sub(r"\s+", " ", b).strip()
# #         if 40 < len(b) < 1500:
# #             if total + len(b) > max_chars:
# #                 break
# #             out.append(b)
# #             total += len(b)
# #     return out


# # # ==============================================================
# # # PROMPT CONSTRUCTION
# # # ==============================================================
# # def build_prompt(question, context, conversation_context="", mode="auto"):
# #     docs = "\n\n".join(
# #         [f"[Document {i+1}]\n{c.strip()}" for i, c in enumerate(context[:8])]
# #     )

# #     system_instr = (
# #         "You are a Syracuse University research assistant. "
# #         "Use the verified research context below to answer factually and clearly. "
# #         "These papers are Syracuse-affiliated or collaborator works. "
# #         "Summarize relevant researchers and findings that appear. "
# #         "Only say 'no relevant information found' if the context is empty."
# #     )
# #     if mode == "abstracts":
# #         system_instr += " (The context contains short abstracts — keep it concise.)"
# #     else:
# #         system_instr += " (The context contains full papers — include specific details.)"

# #     if conversation_context:
# #         system_instr += (
# #             f"\n\nPrevious conversation (for continuity, not evidence):\n{conversation_context}"
# #         )

# #     return (
# #         f"{system_instr}\n\n"
# #         f"Question: {question.strip()}\n\n"
# #         f"Research Context:\n{docs}\n\n"
# #         f"Answer clearly and factually:\n"
# #     )


# # # ==============================================================
# # # MAIN PIPELINE
# # # ==============================================================
# # def answer_question(question, n_ctx=6, use_cache=True, use_conversation=True):
# #     db_conf = get_active_db_config()
# #     db_name = get_db_manager().active_config_name
# #     cache_hit = False

# #     # --- Cache
# #     retr = retrieval_cache.get(question, db_name, n_ctx) if use_cache else None
# #     if retr:
# #         cache_hit = True
# #     else:
# #         retr = hybrid_search(question, k_chroma=n_ctx, k_graph=n_ctx)
# #         if use_cache:
# #             retrieval_cache.set(question, db_name, n_ctx, retr)

# #     ctx = sanitize(retr.get("fused_text_blocks", []))
# #     if not ctx:
# #         return {
# #             "answer": "No research context available.",
# #             "sources": [],
# #             "graph_hits": [],
# #             "cache_hit": cache_hit,
# #             "db_config": db_name,
# #         }

# #     conv_ctx = conversation_memory.get_context() if use_conversation else ""
# #     mode = "abstracts" if "abstract" in db_name.lower() else "full"

# #     prompt = build_prompt(question, ctx, conv_ctx, mode)
# #     tok, model = _load_llm()
# #     inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2000)
# #     inputs = {k: v.to(model.device) for k, v in inputs.items()}

# #     with torch.no_grad():
# #         gen = model.generate(
# #             **inputs,
# #             max_new_tokens=600,
# #             temperature=0.3,
# #             top_p=0.9,
# #             do_sample=False,
# #             repetition_penalty=1.1,
# #             pad_token_id=tok.pad_token_id,
# #             eos_token_id=tok.eos_token_id,
# #         )
# #     out_text = tok.decode(gen[0], skip_special_tokens=True)
# #     ans = out_text.split("Answer clearly and factually:")[-1].strip()

# #     # --- Fallback
# #     if ans.lower().startswith("no relevant") or len(ans) < 40:
# #         ans = (
# #             "Based on the retrieved Syracuse research, the following is relevant:\n\n"
# #             + "\n\n".join(ctx[:2])
# #         )

# #     if use_conversation:
# #         conversation_memory.add_message("user", question)
# #         conversation_memory.add_message("assistant", ans)

# #     return {
# #         "answer": ans,
# #         "sources": ctx,
# #         "graph_hits": retr.get("graph_hits", []),
# #         "cache_hit": cache_hit,
# #         "db_config": db_name,
# #     }


# # # ==============================================================
# # # UTILITIES
# # # ==============================================================
# # def clear_cache():
# #     retrieval_cache.clear()
# #     print("🗑️ Cache cleared")


# # def clear_conversation():
# #     conversation_memory.clear()
# #     print("🗑️ Conversation memory cleared")


# # def get_cache_stats():
# #     return retrieval_cache.stats()


# # def get_conversation_summary():
# #     return {
# #         "buffer_size": len(conversation_memory.buffer),
# #         "archived": conversation_memory.archived,
# #         "has_summary": bool(conversation_memory.summary),
# #     }


# # # Pre-warm model
# # try:
# #     _load_llm()
# # except Exception as e:
# #     print(f"⚠️ Model pre-warm skipped: {e}")


# """
# rag_pipeline.py — SyracuseRAG-LLaMA Final Stable (Nov 2025)
# - Sorting happens in DB layer (database_manager / retriever)
# - Prompt explicitly tells LLM that context is newest → oldest
# - Streamlit keys kept: cache stats + conversation summary
# """
# import re, torch, hashlib
# from collections import deque
# from datetime import datetime
# from transformers import AutoTokenizer, AutoModelForCausalLM

# import config_full as config
# from hybrid_langchain_retriever import hybrid_search
# from database_manager import get_active_db_config, get_db_manager


# # =========================
# # DEVICE + MODEL
# # =========================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# dtype  = torch.float16 if device == "cuda" else torch.float32
# tok, model = None, None


# def _load_llm():
#     global tok, model
#     if tok and model:
#         return tok, model
#     print(f"⚙️  Loading LLaMA model from {config.LLAMA_MODEL_PATH} …")
#     tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH, use_fast=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         config.LLAMA_MODEL_PATH,
#         dtype=dtype,
#         device_map="auto" if device == "cuda" else None,
#         low_cpu_mem_usage=True,
#     )
#     if tok.pad_token_id is None and tok.eos_token_id:
#         tok.pad_token_id = tok.eos_token_id
#     print(f"✅ Model loaded on {device}")
#     return tok, model


# # =========================
# # MEMORY + CACHE
# # =========================
# class ConversationMemory:
#     def __init__(self, buffer_size=4):
#         self.buffer = deque(maxlen=buffer_size)
#         self.summary = ""
#         self.archived = 0

#     def add_message(self, role, content):
#         self.buffer.append(
#             {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
#         )
#         if len(self.buffer) >= self.buffer.maxlen:
#             self._summarize_partial()

#     def _summarize_partial(self):
#         if self.archived >= 20 or len(self.buffer) < 2:
#             return
#         msgs = [self.buffer.popleft() for _ in range(2)]
#         snippet = "\n".join([f"{m['role']}: {m['content'][:120]}…" for m in msgs])
#         self.summary = f"{self.summary}\n[Earlier] {snippet}" if self.summary else snippet
#         self.archived += 2

#     def get_context(self, max_chars=600):
#         parts = []
#         if self.summary:
#             parts.append(self.summary[: max_chars // 2])
#         for m in list(self.buffer)[-3:]:
#             parts.append(f"{m['role'].capitalize()}: {m['content'][:200]}")
#         return "\n".join(parts)

#     def clear(self):
#         self.buffer.clear()
#         self.summary = ""
#         self.archived = 0


# class RetrievalCache:
#     def __init__(self, maxsize=100):
#         self.cache, self.order = {}, deque(maxlen=maxsize)
#         self.maxsize = maxsize
#         self.hits = self.misses = 0

#     def _key(self, q, db, n):
#         return hashlib.sha256(f"{q.lower()}_{db}_{n}".encode()).hexdigest()[:16]

#     def get(self, q, db, n):
#         k = self._key(q, db, n)
#         if k in self.cache:
#             self.hits += 1
#             self.order.remove(k)
#             self.order.append(k)
#             return self.cache[k]
#         self.misses += 1
#         return None

#     def set(self, q, db, n, val):
#         k = self._key(q, db, n)
#         if len(self.cache) >= self.maxsize and k not in self.cache:
#             old = self.order.popleft()
#             del self.cache[old]
#         self.cache[k] = val
#         if k in self.order:
#             self.order.remove(k)
#         self.order.append(k)

#     def stats(self):
#         total = self.hits + self.misses
#         hit_rate = f"{(self.hits / total * 100):.1f}%" if total else "0%"
#         return {
#             "size": len(self.cache),
#             "maxsize": self.maxsize,
#             "hits": self.hits,
#             "misses": self.misses,
#             "hit_rate": hit_rate,
#         }

#     def clear(self):
#         self.cache.clear()
#         self.order.clear()
#         self.hits = self.misses = 0


# retrieval_cache = RetrievalCache()
# conversation_memory = ConversationMemory()


# # =========================
# # SANITIZE + PROMPT
# # =========================
# def sanitize(blocks, max_chars=6000):
#     out, total = [], 0
#     for b in blocks:
#         b = re.sub(r"<[^>]+>", " ", str(b))
#         b = re.sub(r"\s+", " ", b).strip()
#         if 40 < len(b) < 1500:
#             if total + len(b) > max_chars:
#                 break
#             out.append(b)
#             total += len(b)
#     return out


# def build_prompt(question, context, conversation_context="", mode="auto"):
#     docs = "\n\n".join(
#         [f"[Document {i+1} — Most Recent First]\n{c.strip()}" for i, c in enumerate(context[:8])]
#     )

#     system_instr = (
#         "You are a Syracuse University research assistant. "
#         "Use only the Syracuse-affiliated research context below. "
#         "The context is sorted newest → oldest. "
#         "Prioritize insights from the latest studies; use earlier work for background only. "
#         "Do not speculate or use information outside this context."
#     )

#     if mode == "abstracts":
#         system_instr += (
#             " The context contains short abstracts — highlight main findings and note uncertainty if details are missing."
#         )
#     else:
#         system_instr += (
#             " The context contains full papers — include methods, results, and collaborators where relevant."
#         )

#     if conversation_context:
#         system_instr += f"\n\nPrevious conversation (for continuity, not evidence):\n{conversation_context}"

#     return (
#         f"{system_instr}\n\n"
#         f"Question: {question.strip()}\n\n"
#         f"Research Context (newest → oldest):\n{docs}\n\n"
#         f"Answer factually, prioritizing the latest research:\n"
#     )


# # =========================
# # MAIN
# # =========================
# def answer_question(question, n_ctx=6, use_cache=True, use_conversation=True):
#     db_conf = get_active_db_config()
#     db_name = get_db_manager().active_config_name
#     cache_hit = False

#     retr = retrieval_cache.get(question, db_name, n_ctx) if use_cache else None
#     if retr:
#         cache_hit = True
#     else:
#         retr = hybrid_search(question, k_chroma=n_ctx, k_graph=n_ctx)
#         if use_cache:
#             retrieval_cache.set(question, db_name, n_ctx, retr)

#     ctx = sanitize(retr.get("fused_text_blocks", []))
#     if not ctx:
#         return {
#             "answer": "No research context available.",
#             "sources": [],
#             "graph_hits": [],
#             "cache_hit": cache_hit,
#             "db_config": db_name,
#         }

#     conv_ctx = conversation_memory.get_context() if use_conversation else ""
#     mode = "abstracts" if "abstract" in db_name.lower() else "full"

#     prompt = build_prompt(question, ctx, conv_ctx, mode)
#     tok, model = _load_llm()
#     inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2000)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}

#     with torch.no_grad():
#         gen = model.generate(
#             **inputs,
#             max_new_tokens=600,
#             temperature=0.3,
#             top_p=0.9,
#             do_sample=False,
#             repetition_penalty=1.1,
#             pad_token_id=tok.pad_token_id,
#             eos_token_id=tok.eos_token_id,
#         )

#     out_text = tok.decode(gen[0], skip_special_tokens=True)
#     ans = out_text.split("Answer factually, prioritizing the latest research:")[-1].strip()

#     if len(ans) < 40 or ans.lower().startswith("no relevant"):
#         ans = "Based on Syracuse research, key findings include:\n" + "\n\n".join(ctx[:2])

#     if use_conversation:
#         conversation_memory.add_message("user", question)
#         conversation_memory.add_message("assistant", ans)

#     return {
#         "answer": ans,
#         "sources": ctx,
#         "graph_hits": retr.get("graph_hits", []),
#         "cache_hit": cache_hit,
#         "db_config": db_name,
#         "conversation_used": use_conversation
#     }


# # =========================
# # UTILITIES
# # =========================
# def clear_cache():
#     retrieval_cache.clear()
#     print("🗑️ Cache cleared")


# def clear_conversation():
#     conversation_memory.clear()
#     print("🗑️ Conversation memory cleared")


# def get_cache_stats():
#     return retrieval_cache.stats()


# def get_conversation_summary():
#     return {
#         "buffer_size": len(conversation_memory.buffer),
#         "archived_count": conversation_memory.archived,  # Streamlit-compatible key
#         "has_summary": bool(conversation_memory.summary),
#     }


# # Pre-warm
# try:
#     _load_llm()
# except Exception as e:
#     print(f"⚠️ Model pre-warm skipped: {e}")


"""
rag_pipeline.py — SyracuseRAG-LLaMA Final Stable (Nov 2025)
- Sorting happens in DB layer (database_manager / retriever)
- Prompt explicitly tells LLM that context is newest → oldest
- Streamlit keys kept: cache stats + conversation summary
"""
import re, torch, hashlib
from collections import deque
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

import config_full as config
from hybrid_langchain_retriever import hybrid_search
from database_manager import get_active_db_config, get_db_manager


# =========================
# DEVICE + MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
tok, model = None, None


def _load_llm():
    global tok, model
    if tok and model:
        return tok, model
    print(f"⚙️  Loading LLaMA model from {config.LLAMA_MODEL_PATH} …")
    tok = AutoTokenizer.from_pretrained(config.LLAMA_MODEL_PATH, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.LLAMA_MODEL_PATH,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if tok.pad_token_id is None and tok.eos_token_id:
        tok.pad_token_id = tok.eos_token_id
    print(f"✅ Model loaded on {device}")
    return tok, model


# =========================
# MEMORY + CACHE
# =========================
class ConversationMemory:
    def __init__(self, buffer_size=4):
        self.buffer = deque(maxlen=buffer_size)
        self.summary = ""
        self.archived = 0

    def add_message(self, role, content):
        self.buffer.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )
        if len(self.buffer) >= self.buffer.maxlen:
            self._summarize_partial()

    def _summarize_partial(self):
        if self.archived >= 20 or len(self.buffer) < 2:
            return
        msgs = [self.buffer.popleft() for _ in range(2)]
        snippet = "\n".join([f"{m['role']}: {m['content'][:120]}…" for m in msgs])
        self.summary = f"{self.summary}\n[Earlier] {snippet}" if self.summary else snippet
        self.archived += 2

    def get_context(self, max_chars=600):
        parts = []
        if self.summary:
            parts.append(self.summary[: max_chars // 2])
        for m in list(self.buffer)[-3:]:
            parts.append(f"{m['role'].capitalize()}: {m['content'][:200]}")
        return "\n".join(parts)

    def clear(self):
        self.buffer.clear()
        self.summary = ""
        self.archived = 0


class RetrievalCache:
    def __init__(self, maxsize=100):
        self.cache, self.order = {}, deque(maxlen=maxsize)
        self.maxsize = maxsize
        self.hits = self.misses = 0

    def _key(self, q, db, n):
        return hashlib.sha256(f"{q.lower()}_{db}_{n}".encode()).hexdigest()[:16]

    def get(self, q, db, n):
        k = self._key(q, db, n)
        if k in self.cache:
            self.hits += 1
            self.order.remove(k)
            self.order.append(k)
            return self.cache[k]
        self.misses += 1
        return None

    def set(self, q, db, n, val):
        k = self._key(q, db, n)
        if len(self.cache) >= self.maxsize and k not in self.cache:
            old = self.order.popleft()
            del self.cache[old]
        self.cache[k] = val
        if k in self.order:
            self.order.remove(k)
        self.order.append(k)

    def stats(self):
        total = self.hits + self.misses
        hit_rate = f"{(self.hits / total * 100):.1f}%" if total else "0%"
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }

    def clear(self):
        self.cache.clear()
        self.order.clear()
        self.hits = self.misses = 0


retrieval_cache = RetrievalCache()
conversation_memory = ConversationMemory()


# =========================
# SANITIZE + PROMPT
# =========================
def sanitize(blocks, max_chars=6000):
    out, total = [], 0
    for b in blocks:
        b = re.sub(r"<[^>]+>", " ", str(b))
        b = re.sub(r"\s+", " ", b).strip()
        if 40 < len(b) < 1500:
            if total + len(b) > max_chars:
                break
            out.append(b)
            total += len(b)
    return out


def build_prompt(question, context, conversation_context="", mode="auto"):
    docs = "\n\n".join(
        [f"[Document {i+1} — Most Recent First]\n{c.strip()}" for i, c in enumerate(context[:8])]
    )

    system_instr = (
        "You are a Syracuse University research assistant. "
        "Use only the Syracuse-affiliated research context below. "
        "The context is sorted newest → oldest. "
        "Prioritize insights from the latest studies; use earlier work for background only. "
        "Do not speculate or use information outside this context."
    )

    if mode == "abstracts":
        system_instr += (
            " The context contains short abstracts — highlight main findings and note uncertainty if details are missing."
        )
    else:
        system_instr += (
            " The context contains full papers — include methods, results, and collaborators where relevant."
        )

    if conversation_context:
        system_instr += f"\n\nPrevious conversation (for continuity, not evidence):\n{conversation_context}"

    return (
        f"{system_instr}\n\n"
        f"Question: {question.strip()}\n\n"
        f"Research Context (newest → oldest):\n{docs}\n\n"
        f"Answer factually, prioritizing the latest research:\n"
    )


# =========================
# MAIN
# =========================
def answer_question(question, n_ctx=6, use_cache=True, use_conversation=True):
    db_conf = get_active_db_config()
    db_name = get_db_manager().active_config_name
    cache_hit = False

    retr = retrieval_cache.get(question, db_name, n_ctx) if use_cache else None
    if retr:
        cache_hit = True
    else:
        retr = hybrid_search(question, k_chroma=n_ctx, k_graph=n_ctx)
        if use_cache:
            retrieval_cache.set(question, db_name, n_ctx, retr)

    ctx = sanitize(retr.get("fused_text_blocks", []))
    if not ctx:
        return {
            "answer": "No research context available.",
            "sources": [],
            "graph_hits": [],
            "cache_hit": cache_hit,
            "db_config": db_name,
        }

    conv_ctx = conversation_memory.get_context() if use_conversation else ""
    mode = "abstracts" if "abstract" in db_name.lower() else "full"

    prompt = build_prompt(question, ctx, conv_ctx, mode)
    tok, model = _load_llm()
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2000)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.3,
            top_p=0.9,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    out_text = tok.decode(gen[0], skip_special_tokens=True)
    ans = out_text.split("Answer factually, prioritizing the latest research:")[-1].strip()

    if len(ans) < 40 or ans.lower().startswith("no relevant"):
        ans = "Based on Syracuse research, key findings include:\n" + "\n\n".join(ctx[:2])

    if use_conversation:
        conversation_memory.add_message("user", question)
        conversation_memory.add_message("assistant", ans)

    return {
        "answer": ans,
        "sources": ctx,
        "graph_hits": retr.get("graph_hits", []),
        "cache_hit": cache_hit,
        "db_config": db_name,
        "conversation_used": use_conversation
    }


# =========================
# UTILITIES
# =========================
def clear_cache():
    retrieval_cache.clear()
    print("🗑️ Cache cleared")


def clear_conversation():
    conversation_memory.clear()
    print("🗑️ Conversation memory cleared")


def get_cache_stats():
    return retrieval_cache.stats()


def get_conversation_summary():
    return {
        "buffer_size": len(conversation_memory.buffer),
        "archived_count": conversation_memory.archived,  # Streamlit-compatible key
        "has_summary": bool(conversation_memory.summary),
    }


# Pre-warm
try:
    _load_llm()
except Exception as e:
    print(f"⚠️ Model pre-warm skipped: {e}")
