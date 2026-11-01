import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from typing import TypedDict, List

load_dotenv()

FAISS_INDEX_DIR = "faiss_index"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mistral")


def get_llm():
    provider = LLM_PROVIDER.lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"))
    elif provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            st.error("MISTRAL_API_KEY is not set. Add it to .env or set the environment variable.")
            st.stop()
        return ChatMistralAI(
            model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'ollama' or 'mistral'.")


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_data_from_directory():
    docs = []
    data_dir = "data"
    if not os.path.isdir(data_dir):
        return docs
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.endswith(".txt"):
            loader = TextLoader(fpath)
            docs.extend(loader.load())
        elif fname.endswith(".pdf"):
            loader = PyPDFLoader(fpath)
            docs.extend(loader.load())
    return docs


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def load_or_build_vector_store(docs, embeddings):
    if os.path.isdir(FAISS_INDEX_DIR) and os.listdir(FAISS_INDEX_DIR):
        db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        if docs:
            db.add_documents(docs)
            db.save_local(FAISS_INDEX_DIR)
        return db
    if not docs:
        return None
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(FAISS_INDEX_DIR)
    return db


def handle_uploaded_files(uploaded_files):
    new_docs = []
    os.makedirs("data", exist_ok=True)
    for uploaded in uploaded_files:
        dest = os.path.join("data", uploaded.name)
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        if uploaded.name.endswith(".txt"):
            loader = TextLoader(dest)
            new_docs.extend(loader.load())
        elif uploaded.name.endswith(".pdf"):
            loader = PyPDFLoader(dest)
            new_docs.extend(loader.load())
    return new_docs


# --- LangGraph RAG workflow ---

class GraphState(TypedDict):
    question: str
    generation: str
    documents: list
    sources: list


def build_rag_graph(retriever):
    def retrieve_documents(state):
        question = state["question"]
        docs = retriever.invoke(question)
        return {"documents": docs, "question": question}

    def generate_answer(state):
        question = state["question"]
        documents = state["documents"]

        context_parts = []
        sources = []
        for i, doc in enumerate(documents):
            source_name = doc.metadata.get("source", "unknown")
            context_parts.append(doc.page_content)
            sources.append(f"[{i+1}] {os.path.basename(source_name)} (chunk {i+1})")

        context_text = "\n\n".join(context_parts)

        prompt = PromptTemplate(
            template=(
                "You are a helpful assistant. Use ONLY the following context to answer the question.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )

        llm = get_llm()
        rag_chain = prompt | llm | StrOutputParser()
        generation = rag_chain.invoke({"context": context_text, "question": question})
        return {"documents": documents, "question": question, "generation": generation, "sources": sources}

    def no_docs_answer(state):
        return {
            "question": state["question"],
            "generation": "I don't know based on the documents.",
            "documents": [],
            "sources": [],
        }

    def decide_to_generate(state):
        if state["documents"]:
            return "generate"
        return "no_docs"

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("no_docs", no_docs_answer)
    workflow.set_entry_point("retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        decide_to_generate,
        {"generate": "generate", "no_docs": "no_docs"},
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("no_docs", END)
    return workflow.compile()


# --- Streamlit app ---

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    st.title("RAG Chatbot")
    st.caption(f"LLM provider: **{LLM_PROVIDER}**")

    # Sidebar: file uploader
    with st.sidebar:
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Upload .txt or .pdf files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
        )

    embeddings = get_embeddings()

    # Load existing docs from data/
    base_docs = load_data_from_directory()
    base_chunks = split_documents(base_docs)

    # Handle uploaded files
    upload_chunks = []
    if uploaded_files:
        upload_docs = handle_uploaded_files(uploaded_files)
        upload_chunks = split_documents(upload_docs)

    all_chunks = base_chunks + upload_chunks
    db = load_or_build_vector_store(all_chunks, embeddings)

    if db is None:
        st.warning("No documents loaded. Add .txt or .pdf files to the data/ folder or upload them in the sidebar.")
        return

    retriever = db.as_retriever(search_kwargs={"k": 4})
    rag_app = build_rag_graph(retriever)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            result = rag_app.invoke({"question": prompt})

        answer = result.get("generation", "I don't know based on the documents.")
        sources = result.get("sources", [])

        response_text = answer
        if sources:
            response_text += "\n\n---\n**Sources:**\n" + "\n".join(sources)

        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
