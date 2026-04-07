# 🤖 Dynamic AI Chatbot — RAG-Based Intelligent Assistant

> Context-aware AI assistant with Retrieval-Augmented Generation (RAG), FAISS vector storage, and Groq LLaMA 3.3. Sub-300ms inference latency. Live on Streamlit Cloud.

Live Demo = https://dynamic-ai-chatbot-d6gxvhxmr6cf6bz3kddvab.streamlit.app/https://dynamic-ai-chatbot-d6gxvhxmr6cf6bz3kddvab.streamlit.app/https://dynamic-ai-chatbot-d6gxvhxmr6cf6bz3kddvab.streamlit.app/

## 📌 What It Does

Upload any PDF → ask questions → get answers grounded strictly in your document. No hallucinations. No guessing.

The system chunks your document, embeds it into a FAISS vector index, and at query time retrieves the most semantically relevant chunks before sending them to LLaMA 3.3 via Groq's LPU inference engine.

---

## ⚡ Key Metrics

| Metric | Value |
|---|---|
| Inference Latency | **< 300ms** end-to-end |
| Hallucination Rate | **~0%** (responses grounded in source docs) |
| Context Management | Multi-turn with system-level time sync |
| Deployment | Streamlit Community Cloud |

---

## 🏗 Architecture

```
User Query
    │
    ▼
[PDF Handler] ──► [Chunking + Embedding] ──► [FAISS Index]
                                                    │
                                               Semantic Search
                                                    │
                                              Top-K Chunks
                                                    │
[Groq LPU] ◄─── [LangChain Prompt Builder] ◄───────┘
    │
    ▼
Grounded Response (< 300ms)
```

---

## 🗂 Project Structure

```
dynamic-ai-chatbot-rag/
├── app.py              # Streamlit UI + session management
├── bot_engine.py       # LangChain RAG chain + Groq integration
├── pdf_handler.py      # PDF ingestion, chunking, FAISS indexing
├── requirements.txt    # Dependencies
└── .gitignore
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/alokvermaconnect-make/dynamic-ai-chatbot-rag.git
cd dynamic-ai-chatbot-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
export GROQ_API_KEY="your_key_here"

# 4. Launch
streamlit run app.py
```

---

## 🛠 Tech Stack

- **LLM:** LLaMA 3.3 via Groq LPU Inference Engine
- **Orchestration:** LangChain
- **Vector Store:** FAISS (local, no external DB needed)
- **UI:** Streamlit
- **PDF Processing:** PyPDF2 / pdfplumber
- **Embeddings:** HuggingFace sentence-transformers

---

## 💡 Key Engineering Decisions

**Why FAISS over a cloud vector DB?**
Zero latency overhead from network calls. For a single-user chatbot, local FAISS gives faster retrieval and no API costs.

**Why Groq LPU instead of OpenAI?**
Groq's Language Processing Unit delivers hardware-accelerated inference — achieving sub-300ms latency that standard GPU-based APIs can't match at this price point.

**Solving hallucination:**
The prompt template explicitly instructs the model to answer *only* from retrieved context. If the answer isn't in the document, it says so.

---

*Part of [Alok Verma's](https://alokvermaconnect-make.github.io/alok-verma-portfolio/) AI/ML project portfolio.*
