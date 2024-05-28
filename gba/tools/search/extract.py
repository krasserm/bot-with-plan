from concurrent.futures import ThreadPoolExecutor
from typing import List

from gba.client import Llama3Instruct, ChatClient

EXTRACTOR_CONTEXT_SYSTEM_PROMPT = "You are a helpful assistant that extracts content relevant to a given query only based on the provided context. You omit the existence of the context in your answers."

EXTRACTOR_CONTEXT_USER_PROMPT_TEMPLATE = """Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge extract all information from the context that is relevant to answer the query without reformulating the context.

The response should only contain extracted passages from the context, and should not contain any new information.
Person names in the context information must exactly match person names in the query (firstname and lastname) to be relevant.
If the context does not contain relevant information for the query just output `no information`.

Query: {query_str}
"""


class ContentExtractor:
    def __init__(self, model: Llama3Instruct, max_workers: int = 10):
        self._client = ChatClient(model)
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def extract(self, query: str, documents: List[str], temperature: float = -1) -> List[str]:
        messages = [
            EXTRACTOR_CONTEXT_USER_PROMPT_TEMPLATE.format(context_str=doc, query_str=query) for doc in documents
        ]

        fts = []
        for message in messages:
            fts.append(self._pool.submit(self._extract_blocking, message, temperature))

        results = []
        for ft in fts:
            results.append(ft.result())

        return results

    def _extract_blocking(self, message, temperature):
        messages = [
            {"role": "system", "content": EXTRACTOR_CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]

        response = self._client.complete(messages, temperature=temperature)
        return response["content"].replace("\n", " ").strip()
