import json

from langchain_core.messages import HumanMessage
from langchain_experimental.chat_models.llm_wrapper import ChatWrapper
from pydantic import BaseModel

from gba.tools.base import Tool
from gba.utils import prop_order_from_schema, Scratchpad


class Result(BaseModel):
    summary: str
    answer: str


PROMPT_TEMPLATE = """User request:

```
{request}
```

Context information:

```
{context}
```

Answer the user request using the available context information only but do not mention the existence of that context information. Use the following output format:

{{
  "summary": <summarize the request and all results from the context information>,
  "answer": <your detailed answer to the user request. Only answer what has been actually requested>
}}

Never make your own calculations because you are bad at math."""


class RespondTool(Tool):
    name: str = "respond_to_user"

    def __init__(self, model: ChatWrapper):
        self.model = model

    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str:
        """Useful for responding with a final answer to the user request."""
        if scratchpad.is_empty():
            context_str = task
        else:
            context_str = "\n\n".join(str(entry) for entry in scratchpad.entries)

        prompt = PROMPT_TEMPLATE.format(request=request, context=context_str)
        schema = Result.model_json_schema()

        ai_message = self.model.invoke(
            input=[HumanMessage(content=prompt)],
            schema=schema,
            prop_order=prop_order_from_schema(schema),
            prompt_ext=self.model.ai_n_beg,
        )

        result = json.loads(ai_message.content)
        return result["answer"]
