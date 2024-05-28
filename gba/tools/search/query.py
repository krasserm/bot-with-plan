from gba.client import Llama3Instruct, ChatClient

REWRITE_QUERY_SYSTEM_PROMPT = "You are a helpful assistant that converts a task description into a search query in natural language for performing a web search."

REWRITE_QUERY_USER_PROMPT_TEMPLATE = """Convert the following task description into an search query for a web search.
Follow these rules:
* Only output the query.{extra}
* Omit the terms "Wikipedia", "search" or "site:..." in the query.
* Only use information from the task description to construct the query.
* Do not use prior knowledge to construct the query.
* Keep all relevant keywords from the task description in the query.
* Always retain cardinal and ordinal information from the task description in the query (e.g. 'first', 'second', ...)

Task: {task}
"""


class QueryRewriter:
    def __init__(self, llm: Llama3Instruct):
        self._client = ChatClient(llm)

    def rewrite(self, task: str, temperature: float = -1, natural_language: bool = True) -> str:
        extra = "\n* Only use natural language." if natural_language else ""
        message = REWRITE_QUERY_USER_PROMPT_TEMPLATE.format(task=task, extra=extra)

        messages = [
            {"role": "system", "content": REWRITE_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]

        response = self._client.complete(messages, temperature=temperature)
        return response["content"].strip('"')
