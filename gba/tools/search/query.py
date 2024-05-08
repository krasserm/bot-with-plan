from langchain_core.messages import SystemMessage, HumanMessage

from gba.client import Llama3Instruct

REWRITE_QUERY_SYSTEM_PROMPT = "You are a helpful assistant that converts a task description into a search query in natural language for performing a web search."

REWRITE_QUERY_USER_PROMPT_TEMPLATE = """Convert the following task description into an search query for a web search.
Follow these rules:
* Only output the query.
* Only use natural language.
* Omit the terms "Wikipedia", "search" or "site:..." in the query.
* Only use information from the task description to construct the query.
* Do not use prior knowledge to construct the query.
* Keep all relevant keywords from the task description in the query.
* Always retain cardinal and ordinal information from the task description in the query (e.g. 'first', 'second', ...)

Task: {task}
"""


class QueryRewriter:
    def __init__(self, model: Llama3Instruct):
        self._model = model

    def rewrite(self, task: str):
        message = REWRITE_QUERY_USER_PROMPT_TEMPLATE.format(task=task)

        response = self._model.invoke(
            input=[
                SystemMessage(content=REWRITE_QUERY_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ],
            prompt_ext=self._model.ai_n_beg,
        )
        return response.content.strip('"')
