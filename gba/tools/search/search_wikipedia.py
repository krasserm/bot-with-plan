import logging
import re
import shutil
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download, cached_assets_path
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.quantization import quantize_embeddings
from sqlitedict import SqliteDict
from usearch.index import Index

from gba.client import Llama3Instruct
from gba.tools import Tool
from gba.tools.search.extract import ContentExtractor
from gba.tools.search.query import QueryRewriter
from gba.utils import Scratchpad, StopWatch, recombine_files

logger = logging.getLogger(__name__)


ARTIFACT_REPO_ID = "krasserm/wikipedia-2023-11-en-index"

QA_SYSTEM_PROMPT = "You are a question answering assistant that answers questions only based on the provided context. You omit the existence of the context in your answers."

QA_USER_PROMPT_TEMPLATE = """Context information is below. Each line is a separate document about a specific topic or person.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge precisely answer the query.

Use only information from the context.
To use information from the context related to a person ensure that the person's firstname and lastname in the context matches exactly the person's firstname and lastname in the query.
The answer should be a single sentence and contain the relevant information from the context to answer the query.
If the context does not provide the information that is requested say "No information found".

Query: "{query_str}"
"""


class SearchWikipediaTool(Tool):
    name = "search_wikipedia"

    def __init__(
        self,
        llm_model: Llama3Instruct,
        embedding_model: SentenceTransformer,
        rerank_model: CrossEncoder,
        top_k_nodes: int = 10,
        top_k_related_documents: int = 1,
        top_k_related_nodes: int = 3,
        similarity_search_top_k_nodes: int = 100,
        similarity_search_rescore_multiplier: int = 4,
        extractor: ContentExtractor | None = None,
        cache_dir: Path | None = None,
    ):
        """Search Wikipedia for information.

        :param llm_model: The LLM model to use for generating responses.
        :param embedding_model: SentenceTransformer model used for similarity search.
        :param rerank_model: The reranker model to use for reranking document nodes.
        :param top_k_nodes: Number of top nodes to use in response generation.
        :param top_k_related_documents: Number of related documents to search for related nodes.
        :param top_k_related_nodes: Number of related nodes to use in response generation.
        :param similarity_search_top_k_nodes: Number of top nodes to use in similarity search.
        :param similarity_search_rescore_multiplier: Rescore multiplier for similarity search.
        :param extractor: The content extractor to use for extracting relevant information.
        :param cache_dir: The cache directory to use for downloaded artifacts. By default the huggingface cache directory is used.
        """
        self._top_k_nodes = top_k_nodes
        self._top_k_related_documents = top_k_related_documents
        self._top_k_related_nodes = top_k_related_nodes
        self._similarity_search_top_k_nodes = similarity_search_top_k_nodes
        self._similarity_search_rescore_multiplier = similarity_search_rescore_multiplier

        self._llm_model = llm_model
        self._embedding_model = embedding_model
        self._reranker = rerank_model
        self._query_rewriter = QueryRewriter(model=self._llm_model)
        self._extractor = extractor

        self._synthesise_query_pattern = re.compile(r"^(search (for|to))|(search wikipedia (for|to))\s", re.IGNORECASE)

        self._int8_index_view = self._load_int8_index(cache_dir=cache_dir)
        self._document_mapping = self._load_document_mapping(cache_dir=cache_dir)
        self._binary_index = self._load_binary_index(cache_dir=cache_dir)
        self._dataset = load_dataset("krasserm/wikipedia-2023-11-en-text", split="train", cache_dir=cache_dir)

    @staticmethod
    def _load_document_mapping(cache_dir: Path | None = None):
        logger.warning("Loading document mapping...")
        url_mapping_path = hf_hub_download(
            ARTIFACT_REPO_ID, repo_type="dataset", filename="document-url-mappings.sqlite", cache_dir=cache_dir
        )
        return SqliteDict(url_mapping_path)

    @staticmethod
    def _load_binary_index(cache_dir: Path | None = None) -> faiss.IndexBinaryFlat:
        logger.warning("Loading binary search index...")
        binary_index_path = hf_hub_download(
            ARTIFACT_REPO_ID, repo_type="dataset", filename="faiss-ubinary.index", cache_dir=cache_dir
        )
        return faiss.read_index_binary(binary_index_path)

    @staticmethod
    def _load_int8_index(cache_dir: Path | None = None):
        logger.warning("Loading int8 search index...")
        int8_index_dir = (
            cached_assets_path(library_name="gba", namespace="wikipedia-2023-11-en-index")
            if not cache_dir
            else cache_dir
        )
        int8_index_path = Path(int8_index_dir) / "usearch-int8.index"
        if not int8_index_path.exists():
            logger.warning("Downloading int8 search index parts...")
            int8_index_part_path = None
            for i in range(1, 11):
                int8_index_part_path = hf_hub_download(
                    ARTIFACT_REPO_ID,
                    repo_type="dataset",
                    filename=f"00{i:02}.part",
                    subfolder="usearch-int8-index",
                    cache_dir=cache_dir,
                )

            if int8_index_part_path is None:
                raise ValueError("Failed to download int8 search index parts")

            int8_index_parts_path = Path(int8_index_part_path).parent

            logger.warning("Combining int8 search index parts...")
            recombine_files(Path(int8_index_parts_path), int8_index_path)

            shutil.rmtree(int8_index_parts_path)

        return Index.restore(int8_index_path, view=True)

    def run(
        self,
        request: str,
        task: str,
        scratchpad: Scratchpad,
        **kwargs,
    ) -> str:
        """Useful for searching factual information in Wikipedia."""

        search_query = self._query_rewriter.rewrite(task)
        synthesise_response_query = self._synthesise_query_pattern.sub("", task.strip())

        logger.warning("Searching wikipedia for query '%s'", search_query)

        search_scores, search_indices = self._search(
            query=search_query,
            top_k=self._similarity_search_top_k_nodes,
            rescore_multiplier=self._similarity_search_rescore_multiplier,
        )

        if logger.getEffectiveLevel() <= logging.INFO:
            logger.info("Search results:")
            for score, index in zip(search_scores, search_indices):
                document = self._dataset[index]
                logger.info("(Score: %.4f) %s | %s | %s", score, document["title"], document["url"], document["text"])

        with StopWatch() as sw:
            rerank_scores, rerank_indices = self._rerank(
                query=search_query,
                documents=[self._dataset[idx]["text"] for idx in search_indices],  # type: ignore
                top_k=self._top_k_nodes,
            )
            reranked_node_indices = np.array(search_indices)[rerank_indices].tolist()

        logger.info("Rerank time: %.2f ms", sw.elapsed())

        context_titles, context_documents = self._get_context_documents(
            query=search_query,
            node_scores=rerank_scores,
            node_indices=reranked_node_indices,
            top_k_related_documents=self._top_k_related_documents,
            top_k_related_nodes=self._top_k_related_nodes,
        )

        if self._extractor is not None:
            documents = self._extract_relevant_document_information(
                extractor=self._extractor,
                query=synthesise_response_query,
                titles=context_titles,
                documents=context_documents,
            )
        else:
            documents = list(zip(context_titles, context_documents))

        if not documents:
            return "No information found"

        return self._synthesise_response(synthesise_response_query, documents)

    def _search(self, query: str, top_k: int, rescore_multiplier: int):
        query_embedding = self._embedding_model.encode(
            query,
            normalize_embeddings=True,
            prompt="Represent this sentence for searching relevant passages: ",
        )
        query_embedding_binary = quantize_embeddings(query_embedding.reshape(1, -1), "ubinary")

        _, binary_ids = self._binary_index.search(query_embedding_binary, top_k * rescore_multiplier)  # type: ignore
        binary_ids = binary_ids[0]

        int8_embeddings = self._int8_index_view[binary_ids].astype(int)

        scores = query_embedding @ int8_embeddings.T

        indices = (-scores).argsort()[:top_k]
        top_k_indices = binary_ids[indices]
        top_k_scores = scores[indices]

        return top_k_scores.tolist(), top_k_indices.tolist()

    def _rerank(self, query: str, documents: List[str], top_k: int):
        if not documents:
            return [], []
        results = self._reranker.rank(query, documents, return_documents=False, top_k=top_k)
        return [r["score"] for r in results], [r["corpus_id"] for r in results]

    def _get_context_documents(
        self,
        query: str,
        node_scores: List[float],
        node_indices: List[int],
        top_k_related_documents: int,
        top_k_related_nodes: int,
    ) -> Tuple[List[str], List[str]]:
        wiki_document_urls = []
        wiki_document_titles = []
        wiki_documents = {}

        for score, index in zip(node_scores, node_indices):
            node = self._dataset[index]
            url = node["url"]
            title = node["title"]
            text = node["text"]

            if url not in wiki_documents:
                wiki_document_urls.append(url)
                wiki_document_titles.append(title)
                wiki_documents[url] = [text]
            else:
                wiki_documents[url].append(text)

            logger.info("(Score: %.4f) %s | %s | %s", score, node["title"], node["url"], node["text"])

        related_node_indices = []
        related_nodes = []
        for url in wiki_document_urls[:top_k_related_documents]:
            wiki_document_node_indices = sorted(self._document_mapping[url])
            for idx in wiki_document_node_indices:
                if idx in node_indices:
                    continue
                related_node_indices.append(idx)
                related_nodes.append(self._dataset[idx]["text"])

        with StopWatch() as sw:
            rerank_scores, rerank_indices = self._rerank(query, related_nodes, top_k=top_k_related_nodes)  # type: ignore
            logger.info("Rerank related nodes time: %.2f ms", sw.elapsed())

        logger.info("Related nodes:")
        for score, idx in zip(rerank_scores, np.array(related_node_indices)[rerank_indices].tolist()):
            related_node = self._dataset[idx]
            url = related_node["url"]

            logger.info(
                "(Score: %.4f) %s | %s | %s",
                score,
                related_node["title"],
                related_node["url"],
                related_node["text"],
            )

            if url in wiki_documents:
                wiki_documents[url].append(related_node["text"])
            else:
                continue

        context_titles = []
        context_documents = []
        for title, (_, nodes) in zip(wiki_document_titles, wiki_documents.items()):
            context_titles.append(title)
            context_documents.append(" ".join(nodes))

        return context_titles, context_documents

    def _extract_relevant_document_information(
        self, extractor: ContentExtractor, query: str, titles: List[str], documents: List[str]
    ) -> List[Tuple[str, str]]:
        extracted_documents = extractor.extract(
            query=query,
            documents=[f"{title} - {content}" for title, content in (zip(titles, documents))],
        )

        return [
            (title, document)
            for title, document in zip(titles, extracted_documents)
            if document.strip() != "no information"
        ]

    def _synthesise_response(self, query: str, documents: List[Tuple[str, str]]):
        context = "\n".join([f"{title} - {text}" for title, text in documents])
        message = QA_USER_PROMPT_TEMPLATE.format(
            context_str=context,
            query_str=query,
        )

        logger.info("Prompt:")
        logger.info(message)

        response = self._llm_model.invoke(
            input=[
                SystemMessage(content=QA_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ],
            prompt_ext=self._llm_model.ai_n_beg,
        )
        return response.content
