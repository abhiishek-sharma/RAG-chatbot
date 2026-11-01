"""
Evaluation script for the RAG chatbot.
Tests retrieval hit rate and answer grounding against sample Q&A pairs.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

EVAL_QA_PAIRS = [
    {
        "question": "When was the term artificial intelligence coined?",
        "expected_keywords": ["1956", "john mccarthy", "dartmouth"],
        "expected_source": "ai_history.txt",
    },
    {
        "question": "What happened during the AI winter?",
        "expected_keywords": ["1970s", "1980s", "slowed", "limited computing"],
        "expected_source": "ai_history.txt",
    },
    {
        "question": "What are the three main types of machine learning?",
        "expected_keywords": ["supervised", "unsupervised", "reinforcement"],
        "expected_source": "machine_learning_basics.txt",
    },
    {
        "question": "What is deep learning?",
        "expected_keywords": ["neural networks", "multiple layers"],
        "expected_source": "machine_learning_basics.txt",
    },
    {
        "question": "What are CNNs used for?",
        "expected_keywords": ["image recognition", "convolutional"],
        "expected_source": "machine_learning_basics.txt",
    },
    {
        "question": "What paper introduced the transformer architecture?",
        "expected_keywords": ["attention is all you need", "2017"],
        "expected_source": "machine_learning_basics.txt",
    },
    {
        "question": "Who funded early AI research?",
        "expected_keywords": ["department of defense", "military"],
        "expected_source": "ai_history.txt",
    },
    {
        "question": "What is reinforcement learning?",
        "expected_keywords": ["agent", "rewards", "environment"],
        "expected_source": "machine_learning_basics.txt",
    },
    {
        "question": "What is K-means?",
        "expected_keywords": ["clustering", "unsupervised"],
        "expected_source": "machine_learning_basics.txt",
    },
    {
        "question": "When did AI experience a resurgence?",
        "expected_keywords": ["1990s", "2000s", "powerful computers"],
        "expected_source": "ai_history.txt",
    },
]


def load_all_docs():
    docs = []
    data_dir = "data"
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if fname.endswith(".txt"):
            docs.extend(TextLoader(fpath).load())
        elif fname.endswith(".pdf"):
            docs.extend(PyPDFLoader(fpath).load())
    return docs


def build_index(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db


def evaluate_retrieval(db):
    print("=" * 60)
    print("RAG CHATBOT EVALUATION")
    print("=" * 60)

    retrieval_hits = 0
    grounding_hits = 0
    total = len(EVAL_QA_PAIRS)

    for i, qa in enumerate(EVAL_QA_PAIRS, 1):
        question = qa["question"]
        expected_keywords = qa["expected_keywords"]
        expected_source = qa["expected_source"]

        results = db.similarity_search(question, k=4)
        retrieved_text = " ".join(r.page_content.lower() for r in results)
        retrieved_sources = [os.path.basename(r.metadata.get("source", "")) for r in results]

        source_hit = expected_source in retrieved_sources
        keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in retrieved_text)
        keyword_ratio = keyword_hits / len(expected_keywords) if expected_keywords else 0
        grounded = keyword_ratio >= 0.5

        if source_hit:
            retrieval_hits += 1
        if grounded:
            grounding_hits += 1

        status_retrieval = "HIT" if source_hit else "MISS"
        status_grounding = "GROUNDED" if grounded else "NOT GROUNDED"

        print(f"\n[{i}/{total}] {question}")
        print(f"  Retrieval: {status_retrieval} (expected: {expected_source})")
        print(f"  Sources:   {retrieved_sources}")
        print(f"  Grounding: {status_grounding} ({keyword_hits}/{len(expected_keywords)} keywords found)")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Retrieval hit rate:  {retrieval_hits}/{total} ({retrieval_hits/total*100:.0f}%)")
    print(f"Grounding hit rate:  {grounding_hits}/{total} ({grounding_hits/total*100:.0f}%)")
    print("=" * 60)

    return retrieval_hits, grounding_hits, total


def main():
    print("Loading documents from data/ ...")
    docs = load_all_docs()
    if not docs:
        print("ERROR: No documents found in data/")
        sys.exit(1)
    print(f"Loaded {len(docs)} documents.")

    print("Building FAISS index ...")
    db = build_index(docs)

    retrieval_hits, grounding_hits, total = evaluate_retrieval(db)

    if retrieval_hits < total * 0.7:
        print("\nWARNING: Retrieval hit rate below 70%")
        sys.exit(1)


if __name__ == "__main__":
    main()
