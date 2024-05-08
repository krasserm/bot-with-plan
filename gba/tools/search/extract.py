from concurrent.futures import ThreadPoolExecutor
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from gba.client import Llama3Instruct

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
        self._llm = model
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def extract(self, query: str, documents: List[str]):
        messages = [
            EXTRACTOR_CONTEXT_USER_PROMPT_TEMPLATE.format(context_str=doc, query_str=query) for doc in documents
        ]

        fts = []
        for message in messages:
            fts.append(self._pool.submit(self._extract_blocking, message))

        results = []
        for ft in fts:
            results.append(ft.result())

        return results

    def _extract_blocking(self, message):
        response = self._llm.invoke(
            input=[
                SystemMessage(content=EXTRACTOR_CONTEXT_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ],
            prompt_ext=self._llm.ai_n_beg,
        )
        return response.content.replace("\n", " ").strip()
