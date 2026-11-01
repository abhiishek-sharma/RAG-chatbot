# RAG Chatbot

A Retrieval-Augmented Generation chatbot built with Streamlit, LangChain, LangGraph, FAISS, and HuggingFace embeddings. Ask questions about your documents and get grounded answers with source citations.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ Chat      │  │ File Uploader│  │ Chat History      │ │
│  │ Input     │  │ (sidebar)    │  │ (session state)   │ │
│  └─────┬────┘  └──────┬───────┘  └───────────────────┘ │
└────────┼───────────────┼────────────────────────────────┘
         │               │
         ▼               ▼
┌─────────────────────────────────────────────────────────┐
│               LangGraph RAG Workflow                    │
│                                                         │
│  ┌──────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │ Retrieve  │───▶│ Has docs?   │───▶│ Generate      │  │
│  │ (FAISS)   │    │             │    │ (LLM + prompt)│  │
│  └──────────┘    │  No docs:   │    └───────┬───────┘  │
│                  │  "I don't   │            │          │
│                  │   know"     │            ▼          │
│                  └─────────────┘    ┌───────────────┐  │
│                                     │ Answer +      │  │
│                                     │ Source Citing  │  │
│                                     └───────────────┘  │
└─────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌─────────────────────┐          ┌────────────────────────┐
│ FAISS Vector Store   │          │ LLM Provider           │
│ (persisted to disk)  │          │ ┌────────┐ ┌────────┐ │
│                      │          │ │Mistral │ │Ollama  │ │
│ HuggingFace          │          │ │ (API)  │ │(local) │ │
│ all-MiniLM-L6-v2     │          │ └────────┘ └────────┘ │
└─────────────────────┘          └────────────────────────┘
```

## Features

- **RAG pipeline** with LangGraph: retrieve → decide → generate
- **Dual LLM support**: Mistral API (default, no local setup) or Ollama (local)
- **Document support**: `.txt` and `.pdf` files from `data/` folder
- **File uploader**: upload documents via the Streamlit sidebar
- **FAISS persistence**: vector index saved to disk, not rebuilt every run
- **Source citations**: each answer lists the file and chunk it came from
- **Graceful fallback**: returns "I don't know based on the documents" when no relevant docs are found

## Setup

### 1. Clone and install

```bash
git clone https://github.com/abhiishek-sharma/RAG-chatbot.git
cd RAG-chatbot
pip install -r requirements.txt
```

### 2. Configure environment

Copy the example and fill in your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your-key-here
```

### 3. Add documents

Place `.txt` or `.pdf` files in the `data/` directory. Two sample files are included.

### 4. Run the app

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Environment Variables

| Variable         | Default                | Description                        |
|------------------|------------------------|------------------------------------|
| `LLM_PROVIDER`   | `mistral`              | `mistral` (API) or `ollama` (local)|
| `MISTRAL_API_KEY` | —                      | Required when provider is `mistral`|
| `MISTRAL_MODEL`   | `mistral-small-latest` | Mistral model to use               |
| `OLLAMA_MODEL`    | `llama3`               | Ollama model to use                |

## Using Ollama (local)

1. Install Ollama: https://ollama.com
2. Pull a model: `ollama pull llama3`
3. Set in `.env`:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3
   ```

## Tests

```bash
python -m pytest tests/ -v
```

Tests cover:
- Document chunking (empty, short, long, overlap, metadata preservation)
- FAISS vector store creation, retrieval, and persistence
- LangGraph RAG flow (with docs → answer, no docs → fallback)
- Data loading from `data/` directory

All LLM calls are mocked — tests run offline.

## Evaluation

The eval script tests retrieval quality against 10 question/expected-answer pairs on the sample documents:

```bash
python eval.py
```

Sample output:

```
============================================================
RESULTS SUMMARY
============================================================
Retrieval hit rate:  10/10 (100%)
Grounding hit rate:  10/10 (100%)
============================================================
```

The evaluation checks:
- **Retrieval hit rate**: did the correct source file appear in the top-4 results?
- **Grounding hit rate**: do the retrieved chunks contain at least 50% of the expected keywords?

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public or connected to Streamlit Cloud)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo, branch, and `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   LLM_PROVIDER = "mistral"
   MISTRAL_API_KEY = "your-key-here"
   ```
5. Click **Deploy**

> **Note**: The first run downloads the HuggingFace embedding model (~90 MB). Ollama is not available on Streamlit Cloud — use `mistral` as the provider.

## Project Structure

```
├── app.py                          # Main Streamlit app + LangGraph RAG
├── eval.py                         # Evaluation script (retrieval + grounding)
├── requirements.txt                # Pinned dependencies
├── .env.example                    # Environment variable template
├── .gitignore
├── data/
│   ├── ai_history.txt              # Sample: history of AI
│   └── machine_learning_basics.txt # Sample: ML fundamentals
└── tests/
    └── test_chunking_and_retrieval.py  # Unit tests (mocked LLM)
```
