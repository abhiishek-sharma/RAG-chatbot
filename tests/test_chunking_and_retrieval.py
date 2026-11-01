import os
import shutil
import hashlib
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app import split_documents, load_data_from_directory, load_or_build_vector_store, build_rag_graph

FAISS_TEST_DIR = "faiss_index_test"


@pytest.fixture(autouse=True)
def cleanup_test_index():
    yield
    if os.path.isdir(FAISS_TEST_DIR):
        shutil.rmtree(FAISS_TEST_DIR)


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for testing — hashes text to a 384-d vector."""

    def _text_to_vector(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        return rng.randn(384).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._text_to_vector(text)


class TestChunking:
    def test_split_empty_list(self):
        result = split_documents([])
        assert result == []

    def test_split_short_document(self):
        doc = Document(page_content="Short text.", metadata={"source": "test.txt"})
        chunks = split_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."

    def test_split_long_document_creates_multiple_chunks(self):
        long_text = "word " * 500
        doc = Document(page_content=long_text, metadata={"source": "long.txt"})
        chunks = split_documents([doc])
        assert len(chunks) > 1

    def test_chunks_preserve_metadata(self):
        doc = Document(page_content="Some content. " * 100, metadata={"source": "meta.txt", "page": 1})
        chunks = split_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "meta.txt"

    def test_chunk_overlap(self):
        long_text = ("Sentence one. " * 50) + ("Sentence two. " * 50)
        doc = Document(page_content=long_text, metadata={"source": "overlap.txt"})
        chunks = split_documents([doc])
        if len(chunks) >= 2:
            end_of_first = chunks[0].page_content[-50:]
            start_of_second = chunks[1].page_content[:50]
            assert any(
                word in start_of_second for word in end_of_first.split()
            ), "Expected overlap between consecutive chunks"


class TestRetrieval:
    @pytest.fixture
    def sample_docs(self):
        return [
            Document(page_content="Python is a programming language created by Guido van Rossum.", metadata={"source": "python.txt"}),
            Document(page_content="Java was developed by James Gosling at Sun Microsystems.", metadata={"source": "java.txt"}),
            Document(page_content="Rust is a systems programming language focused on safety.", metadata={"source": "rust.txt"}),
        ]

    def test_vector_store_creation(self, sample_docs):
        embeddings = FakeEmbeddings()
        with patch("app.FAISS_INDEX_DIR", FAISS_TEST_DIR):
            db = load_or_build_vector_store(sample_docs, embeddings)
            assert db is not None

    def test_retrieval_returns_docs(self, sample_docs):
        embeddings = FakeEmbeddings()
        with patch("app.FAISS_INDEX_DIR", FAISS_TEST_DIR):
            db = load_or_build_vector_store(sample_docs, embeddings)
            results = db.similarity_search("Python", k=2)
            assert len(results) > 0

    def test_faiss_persistence(self, sample_docs):
        embeddings = FakeEmbeddings()
        with patch("app.FAISS_INDEX_DIR", FAISS_TEST_DIR):
            db1 = load_or_build_vector_store(sample_docs, embeddings)
            assert os.path.isdir(FAISS_TEST_DIR)

            db2 = load_or_build_vector_store([], embeddings)
            results = db2.similarity_search("Python", k=1)
            assert len(results) > 0


class TestRAGGraph:
    def test_graph_returns_answer_with_docs(self):
        mock_doc = Document(page_content="The sky is blue.", metadata={"source": "sky.txt"})
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]

        mock_llm = MagicMock()

        def fake_chain_invoke(inputs):
            return "The sky is blue."

        mock_chain = MagicMock()
        mock_chain.invoke = fake_chain_invoke

        with patch("app.get_llm", return_value=mock_llm):
            with patch("app.PromptTemplate") as mock_pt:
                mock_prompt_inst = MagicMock()
                mock_pt.return_value = mock_prompt_inst

                pipe_step1 = MagicMock()
                mock_prompt_inst.__or__ = MagicMock(return_value=pipe_step1)
                pipe_step1.__or__ = MagicMock(return_value=mock_chain)

                graph = build_rag_graph(mock_retriever)
                result = graph.invoke({"question": "What color is the sky?"})

                assert "generation" in result
                assert result["generation"] == "The sky is blue."
                assert len(result["sources"]) > 0
                assert "sky.txt" in result["sources"][0]

    def test_graph_returns_fallback_when_no_docs(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        graph = build_rag_graph(mock_retriever)
        result = graph.invoke({"question": "Unknown topic"})

        assert result["generation"] == "I don't know based on the documents."
        assert result["sources"] == []


class TestLoadData:
    def test_load_data_reads_txt_files(self):
        docs = load_data_from_directory()
        assert len(docs) > 0
        assert any("artificial intelligence" in d.page_content.lower() for d in docs)
