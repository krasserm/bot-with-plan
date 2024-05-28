import copy
import logging
import re
import urllib
import uuid
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np
import requests
from sentence_transformers import CrossEncoder
from unstructured.partition.html import partition_html

from gba.client import Llama3Instruct, ChatClient
from gba.tools import Tool
from gba.tools.search.extract import ContentExtractor
from gba.tools.search.query import QueryRewriter
from gba.utils import Scratchpad, StopWatch

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = "You are a question answering assistant that answers questions only based on the provided context from the internet. You omit the existence of the context in your answers."

QA_USER_PROMPT_TEMPLATE = """Context information is below. Each line is a separate document from the internet about a specific topic or person.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge precisely solve the given task.

Use only information from the context.
To use information from the context related to a person ensure that the person's firstname and lastname in the context matches exactly the person's firstname and lastname in the task.
The answer should be a single sentence and contain the relevant information from the context to solve the task.
If the context does not provide the information that is requested say "No information found".

Task: "{task_str}"
"""

PAGE_NODE_CATEGORY_SORT_ORDER = {
    "NarrativeText": 0,
    "UncategorizedText": 1,
    "ListItem": 2,
    "Title": 3,
}


class SearchInternetTool(Tool):
    name: str = "search_internet"

    def __init__(
        self,
        llm: Llama3Instruct,
        rerank_model: CrossEncoder,
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
    ):
        """Search the internet for information.

        :param llm: The LLM model to use for generating responses.
        :param rerank_model: The reranker model to use for reranking document nodes.
        :param searxng_endpoint: The endpoint of the searxng service.
        :param top_k_documents: The number of top documents to consider for node extraction.
        :param top_k_nodes_per_document: The number of top nodes to use for response generation.
        :param top_k_snippets: The number of top snippets to use for response generation.
        :param num_nodes_rerank_per_document: The number of nodes to consider for reranking per document.
        :param min_score: The minimum score for reranking nodes.
        :param fetch_webpage_multiplier: The multiplier for the number of webpages to fetch.
        :param fetch_webpage_timeout: The timeout for fetching webpages.
        :param max_concurrent_requests: The maximum number of concurrent requests.
        :param use_extractor: Whether to use a content extractor for retrieved nodes.
        """
        self._llm_client = ChatClient(llm)
        self._reranker = rerank_model
        self._searxng_endpoint = searxng_endpoint.strip("/")
        self._top_k_documents = top_k_documents
        self._top_k_nodes_per_document = top_k_nodes_per_document
        self._num_nodes_rerank_per_document = num_nodes_rerank_per_document
        self._top_k_snippets = top_k_snippets
        self._min_score = min_score
        self._fetch_webpage_multiplier = fetch_webpage_multiplier
        self._fetch_webpage_timeout = fetch_webpage_timeout
        self._extractor = ContentExtractor(model=llm) if use_extractor else None
        self._query_rewriter = QueryRewriter(llm=llm)
        self._query_pool = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        self._extractor_query_pattern = re.compile(
            r"^(search (for|to))|(search the internet (for|to))\s", re.IGNORECASE
        )

    def run(
        self,
        request: str,
        task: str,
        scratchpad: Scratchpad,
        temperature: float = -1,
        **kwargs,
    ) -> str:
        """Useful for searching up-to-date information on the internet."""

        search_query = self._query_rewriter.rewrite(task, natural_language=False)

        logger.warning("Searching the internet for query '%s'", search_query)

        search_results = self._search_internet(search_query)
        logger.info("Found %d search results", len(search_results))

        urls = [r["url"] for r in search_results]
        titles = [r["title"] for r in search_results]
        snippets = [r["content"].strip(" ...") + "." if "content" in r else "" for r in search_results]

        context_titles, context_documents, context_is_snippet = self._get_context_documents(
            query=search_query,
            urls=urls,
            titles=titles,
            snippets=snippets,
            top_k_documents=self._top_k_documents,
            top_k_nodes=self._top_k_nodes_per_document,
            top_k_snippets=self._top_k_snippets,
            num_nodes_rerank=self._num_nodes_rerank_per_document,
            min_score=self._min_score,
        )

        if self._extractor is not None:
            documents = self._extract_relevant_document_information(
                extractor=self._extractor,
                query=self._extractor_query_pattern.sub("", task.strip()),
                titles=context_titles,
                documents=context_documents,
                document_is_snippet=context_is_snippet,
            )
        else:
            documents = list(zip(context_titles, context_documents))

        if not documents:
            return "No information found"

        return self._synthesise_response(task.strip(), documents, temperature)

    def _search_internet(self, query: str) -> List[dict]:
        q = f"!go !ddg !qw {query}"  # use google, duckduckgo, and qwant search engines for better stability of search results
        search_response = requests.get(f"{self._searxng_endpoint}/search?format=json&q={urllib.parse.quote(q)}")  # type: ignore
        search_response.raise_for_status()
        search_results = search_response.json()["results"]
        return search_results

    def _get_context_documents(
        self,
        query: str,
        urls: List[str],
        titles: List[str],
        snippets: List[str],
        top_k_documents: int,
        top_k_nodes: int,
        top_k_snippets: int | None,
        num_nodes_rerank: int,
        min_score: float,
    ) -> Tuple[List[str], List[str], List[bool]]:
        webpages_to_fetch = top_k_documents * self._fetch_webpage_multiplier
        logger.info("Requesting top %d web pages", len(urls[:webpages_to_fetch]))

        with StopWatch() as sw:
            webpages = self._fetch_webpages(urls=urls[:webpages_to_fetch], timeout=self._fetch_webpage_timeout)
            logger.info("Fetched %d web pages in %f ms", len(webpages), sw.elapsed())

        context_titles = []
        context_documents = []
        context_is_snippet = []
        successful_webpages = 0
        for url, title, snippet, full_webpage_nodes in zip(
            urls[:webpages_to_fetch], titles[:webpages_to_fetch], snippets[:webpages_to_fetch], webpages
        ):
            url_id = uuid.uuid5(uuid.NAMESPACE_URL, url).hex[:8]
            if successful_webpages >= top_k_documents:
                break

            if full_webpage_nodes:
                successful_webpages += 1

                webpage_nodes = self._sort_page_nodes(full_webpage_nodes)[:num_nodes_rerank]
                webpage_nodes = [node.text.replace("\n", " ").strip() for node in webpage_nodes]

                logger.info(
                    "[%s] Adding webpage to result set (url='%s', num_nodes=%d)",
                    url_id,
                    url,
                    len(webpage_nodes),
                )
                logger.info("[%s] Title: %s (url='%s')", url_id, title, url)
                logger.info("[%s] Snippet: %s (url='%s')", url_id, snippet, url)

                rerank_result = self._reranker.rank(query, webpage_nodes, return_documents=False, top_k=top_k_nodes)
                rerank_ids = [r["corpus_id"] for r in rerank_result]
                rerank_scores = [r["score"] for r in rerank_result]

                ids = []
                scores = []
                for r_score, r_id in zip(rerank_scores, rerank_ids):
                    if r_score >= min_score:
                        scores.append(r_score)
                        ids.append(r_id)

                top_nodes = np.array(webpage_nodes)[ids].tolist()

                if logger.getEffectiveLevel() <= logging.INFO:
                    logger.info("[%s] Top %d nodes for page '%s':", url_id, top_k_nodes, url)
                    for score, node in zip(scores, top_nodes):
                        logger.info("[%s] %f: %s", url_id, score, node)

                context_titles.append(title)
                context_documents.append(" ".join([snippet] + top_nodes))
                context_is_snippet.append(False)
            else:
                logger.info("[%s] Adding snippet to result set (url='%s')", url_id, url)
                logger.info("[%s] Title: %s (url='%s')", url_id, title, url)
                logger.info("[%s] Snippet: %s (url='%s')", url_id, snippet, url)

                context_titles.append(title)
                context_documents.append(snippet)
                context_is_snippet.append(True)

        if top_k_snippets is not None:
            num_docs = len(context_documents)
            if num_docs < top_k_snippets:
                context_titles += titles[num_docs:top_k_snippets]
                context_documents += snippets[num_docs:top_k_snippets]
                context_is_snippet += [True] * (top_k_snippets - num_docs)

        return context_titles, context_documents, context_is_snippet

    def _fetch_webpages(self, urls: List[str], timeout: float | None = None, ssl_verify: bool = True):
        fts = {}
        for idx, url in enumerate(urls):
            ft = self._query_pool.submit(self._fetch_webpage, url, ssl_verify)
            fts[ft] = idx

        done, not_done = futures.wait(fts.keys(), timeout=timeout)

        webpages = []
        for ft in not_done:
            ft.cancel()
            webpages.append((fts[ft], None))

        for ft in done:
            try:
                result = ft.result()
            except TimeoutError:
                result = None
            webpages.append((fts[ft], result))

        return list(map(lambda x: x[1], sorted(webpages, key=lambda x: x[0])))

    @staticmethod
    def _fetch_webpage(_url: str, ssl_verify: bool):
        try:
            return partition_html(url=_url, ssl_verify=ssl_verify)
        except Exception:
            return None

    @staticmethod
    def _sort_page_nodes(page_nodes):
        nodes = []
        for idx, node in enumerate(page_nodes):
            category_order = PAGE_NODE_CATEGORY_SORT_ORDER.get(node.category, len(PAGE_NODE_CATEGORY_SORT_ORDER))
            nodes.append((category_order, idx, node))

        return [node for _, _, node in sorted(nodes, key=lambda x: (x[0], x[1]))]

    @staticmethod
    def _extract_relevant_document_information(
        extractor: ContentExtractor,
        query: str,
        titles: List[str],
        documents: List[str],
        document_is_snippet: List[bool],
    ) -> List[Tuple[str, str]]:
        extraction_document_indices = []
        extraction_documents = []
        for idx, (title, document, is_snippet) in enumerate(zip(titles, documents, document_is_snippet)):
            if not is_snippet:
                extraction_document_indices.append(idx)
                extraction_documents.append(f"{title} - {document}")

        merged_documents = copy.deepcopy(documents)
        extracted_documents = extractor.extract(query, extraction_documents)
        for idx, extracted_document in zip(extraction_document_indices, extracted_documents):
            merged_documents[idx] = extracted_document

        return [
            (title, document)
            for title, document in zip(titles, merged_documents)
            if document.strip() != "no information"
        ]

    def _synthesise_response(self, task: str, documents: List[Tuple[str, str]], temperature: float) -> str:
        context = "\n".join([f"{title} - {text}" for title, text in documents])
        message = QA_USER_PROMPT_TEMPLATE.format(
            context_str=context,
            task_str=task,
        )

        logger.info("Prompt:")
        logger.info(message)

        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]

        response = self._llm_client.complete(messages, temperature=temperature)
        return response["content"]
