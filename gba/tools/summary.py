import json

from pydantic import BaseModel

from gba.client import ChatClient, Llama3Instruct


class Result(BaseModel):
    response: str


SYSTEM_PROMPT = """You are a helpful assistant."""

USER_PROMPT_TEMPLATE = """You are given a task description and the result of attempting to solve that task.

Task description:

```
{task}
```

Result:

```
{result}
```

Formulate the result as response to the task description.
Only extract from the result what is absolutely necessary.

The response should be a single sentence in natural language.
The response should be as short as possible and only contain what has been requested in the task description.

Use the following output format:

{{
  "response": <generated response>
}}"""


class ResultSummarizer:
    def __init__(self, model: Llama3Instruct):
        self.client = ChatClient(model=model)

    def summarize(self, task: str, result: str, temperature: float = -1) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(task=task, result=result)},
        ]

        message = self.client.complete(
            messages,
            schema=Result.model_json_schema(),
            temperature=temperature,
        )

        content_dict = json.loads(message["content"])
        return content_dict["response"]
