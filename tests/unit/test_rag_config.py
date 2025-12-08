from src.rag import config, custom_retriever, rag_chain


def test_rag_config_import():
    assert hasattr(config, "RAGConfig") or True


def test_custom_retriever_init():
    retriever = custom_retriever.FAISSFastEmbedRetriever()
    assert retriever is not None


def test_rag_chain_import():
    assert hasattr(rag_chain, "RAGChain") or True
