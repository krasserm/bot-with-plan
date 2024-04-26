import json

from pydantic import BaseModel

from gba.client import ChatClient, Llama3Instruct
from gba.tools.base import Tool
from gba.utils import Scratchpad


class Result(BaseModel):
    answer: str


SYSTEM_PROMPT = """You are a helpful assistant."""

USER_PROMPT_TEMPLATE = """You are given a user request and context information:

User request:

```
{request}
```

Context information:

```
{context}
```

Answer the user request using the available context information only. 
The answer should be a single sentence in natural language.
Use the following output format:

{{
  "answer": <generated answer>
}}"""


class RespondTool(Tool):
    name: str = "respond_to_user"

    def __init__(self, model: Llama3Instruct):
        self.client = ChatClient(model=model)

    def run(
            self, 
            request: str, 
            task: str, 
            scratchpad: Scratchpad, 
            temperature: float = -1, 
            return_user_prompt: bool = False, 
            **kwargs,
    ) -> str:
        """Useful for responding with a final answer to the user request."""
                
        user_prompt = USER_PROMPT_TEMPLATE.format(request=request, context=scratchpad.entries_repr())

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        message = self.client.complete(
            messages, 
            schema=Result.model_json_schema(),
            temperature=temperature,
        )

        result = json.loads(message["content"])
        answer = result["answer"]

        if return_user_prompt:
            return answer, user_prompt

        return answer
