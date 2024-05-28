from sentence_transformers import CrossEncoder, SentenceTransformer

from gba.client import Llama3Instruct, LlamaCppClient
from gba.tools.search.search_internet import SearchInternetTool
from gba.tools.search.search_wikipedia import SearchWikipediaTool


def create_search_internet_tool(
    llama3_endpoint: str,
    searxng_endpoint: str,
    top_k_documents: int = 3,
    top_k_nodes_per_document: int = 5,
    top_k_snippets: int | None = None,
    num_nodes_rerank_per_document: int = 100,
    min_score: float = 0.01,
    fetch_webpage_multiplier: int = 2,
    fetch_webpage_timeout: float = 5.0,
    max_concurrent_requests: int = 10,
    use_extractor: bool = True,
    cross_encoder_device: str = "cuda",
) -> SearchInternetTool:
    language_model = Llama3Instruct(llm=LlamaCppClient(url=llama3_endpoint, temperature=-1))
    rerank_model = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1", device=cross_encoder_device)

    return SearchInternetTool(
        llm=language_model,
        rerank_model=rerank_model,
        searxng_endpoint=searxng_endpoint,
        top_k_documents=top_k_documents,
        top_k_nodes_per_document=top_k_nodes_per_document,
        top_k_snippets=top_k_snippets,
        num_nodes_rerank_per_document=num_nodes_rerank_per_document,
        min_score=min_score,
        fetch_webpage_multiplier=fetch_webpage_multiplier,
        fetch_webpage_timeout=fetch_webpage_timeout,
        max_concurrent_requests=max_concurrent_requests,
        use_extractor=use_extractor,
    )


def create_search_wikipedia_tool(
    llama3_endpoint: str,
    top_k_nodes: int = 10,
    top_k_related_documents: int = 1,
    top_k_related_nodes: int = 3,
    similarity_search_top_k_nodes: int = 100,
    similarity_search_rescore_multiplier: int = 4,
    use_extractor: bool = True,
    sentence_transformer_device: str = "cuda",
    cross_encoder_device: str = "cuda",
) -> SearchWikipediaTool:
    language_model = Llama3Instruct(llm=LlamaCppClient(url=llama3_endpoint, temperature=-1))
    embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=sentence_transformer_device)
    rerank_model = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1", device=cross_encoder_device)

    return SearchWikipediaTool(
        llm=language_model,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        top_k_nodes=top_k_nodes,
        top_k_related_documents=top_k_related_documents,
        top_k_related_nodes=top_k_related_nodes,
        similarity_search_top_k_nodes=similarity_search_top_k_nodes,
        similarity_search_rescore_multiplier=similarity_search_rescore_multiplier,
        use_extractor=use_extractor,
    )
