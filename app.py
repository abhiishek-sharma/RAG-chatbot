import streamlit as st
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from typing import TypedDict, List


# Load data from the data directory

@st.cache_resource
def load_data():
    loader = DirectoryLoader('data/', glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    return documents

# Split the documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

# Create embeddings and build the FAISS vector store
@st.cache_resource
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db

# Define the state for the graph
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# Define the nodes for the graph
def retrieve_documents(state):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate_answer(state):
    question = state["question"]
    documents = state["documents"]
    
    # Create a prompt for the LLM
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant. Use the following context to answer the question.
        Context: {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )
    
    # Create the RAG chain using LCEL
    llm = ChatOllama(model="llama3")
    rag_chain = prompt | llm | StrOutputParser()
    
    # Generate the answer
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def decide_to_generate(state):
    if state["documents"]:
        return "generate"
    else:
        return "end"

# Build the graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)
workflow.set_entry_point("retrieve")
workflow.add_conditional_edges(
    "retrieve",
    decide_to_generate,
    {
        "generate": "generate",
        "end": END,
    },
)
workflow.add_edge("generate", END)
app = workflow.compile()

# Main function for the Streamlit app
def main():
    st.title("RAG Chatbot")

    # Load and process data
    documents = load_data()
    docs = split_documents(documents)
    db = create_vector_store(docs)
    global retriever
    retriever = db.as_retriever()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display response
        inputs = {"question": prompt}
        result = app.invoke(inputs)
        response = result["generation"]
        
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
