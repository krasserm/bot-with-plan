import json

from langchain_core.messages import HumanMessage
from langchain_experimental.chat_models.llm_wrapper import ChatWrapper
from pydantic import BaseModel


class Result(BaseModel):
    answer: str


PROMPT_TEMPLATE = """Task:

```
{task}
```

Result:

```
{result}
```

Summarize the result so that it is an answer to the task.

- if you cannot find the answer in the result, say that you don't know the answer.
- if you can answer only partially, answer with those parts available in the result.
- never add additional information to your answer.

Your answer should be a single sentence. Use the following output format:

{{
  "answer": <your answer to the user request>
}}"""


OPTION_1 = "Your answer should be a single sentence"
OPTION_2 = "Your answer should be a single sentence with at most 10 words"


class ResultSummarizer:
    def __init__(self, model: ChatWrapper):
        self.model = model

    def summarize(self, task: str, result: str) -> str:
        user_prompt = PROMPT_TEMPLATE.format(task=task, result=result)

        ai_message = self.model.invoke(
            input=[HumanMessage(content=user_prompt)],
            schema=Result.model_json_schema(),
            prompt_ext=self.model.ai_n_beg,
        )

        result = json.loads(ai_message.content)
        return result["answer"]
