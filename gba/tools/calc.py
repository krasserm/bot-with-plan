from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.chat_models.llm_wrapper import ChatWrapper

from gba.agent import Scratchpad, Tool
from gba.utils import extract_code, exec_code


SYSTEM_PROMPT = """Provide answers in Python wrapped into ```."""


USER_PROMPT_TEMPLATE = '''Create a valid Python script for the following task:

```
{task}
```

Exclusively use this context information as input: 

```
{context}
```

The Python script must calculate a single number and assign it to variable "result". Never print to stdout. Always replace variables in context information with their actual values.'''


class CalculateTool(Tool):
    name: str = "calculate"

    def __init__(self, model: ChatWrapper, summarizer=None):
        self.model = model
        self.summarizer = summarizer

    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str:
        """Useful for evaluating numeric expressions."""

        results = [se.result for se in scratchpad.entries]
        results_str = "\n".join(results)

        if not results_str:
            results_str = "<no context information available>"

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT_TEMPLATE.format(context=results_str, task=task)),
        ]

        ai_message = self.model.invoke(
            input=messages,
            prompt_ext=self.model.ai_n_beg,
        )

        code = extract_code(ai_message.content)
        print(f"```python{code}```")

        result = exec_code(code, result_variable_name="result")
        result = str(result)

        if self.summarizer is not None:
            return self.summarizer.summarize(task, result)
        else:
            return str(result)
