from gba.client import ChatClient, Llama3Instruct
from gba.tools.base import Tool
from gba.utils import exec_code, extract_code, Scratchpad


SYSTEM_PROMPT = """You are a helpful coding assistant. Always format Python code as:

```python 
{code}
```"""

USER_PROMPT_TEMPLATE = """You have access to the following context information:

```
{context}
```

Solve the following task with Python code using context information if needed:

```
{task}
```

Use a variable for each required numeric value from the context.
Assign the result to variable "result".
The result must be either a number or a list of numbers.

Let's write the code step by step."""


class CalculateTool(Tool):
    name: str = "calculate"

    def __init__(self, model: Llama3Instruct, summarizer=None):
        self.client = ChatClient(model=model)
        self.summarizer = summarizer

    def run(
            self, 
            request: str, 
            task: str, 
            scratchpad: Scratchpad, 
            temperature: float = -1, 
            return_user_prompt: bool = False, 
        **kwargs,
    ) -> str:    
        """Useful for evaluating numeric expressions."""

        user_prompt = USER_PROMPT_TEMPLATE.format(context=scratchpad.results_repr(), task=task)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        message = self.client.complete(messages, temperature=temperature)
        code = extract_code(message["content"], remove_print_statements=True)
        
        print(f"```python{code}```")

        result = exec_code(code, result_variable_name="result")        

        if isinstance(result, list):
            result = ", ".join([self.format_number(elem) for elem in result])
        else:
            result = self.format_number(result)

        if self.summarizer is not None:
            result = self.summarizer.summarize(task, result, temperature=temperature)

        if return_user_prompt:
            return result, user_prompt

        return result

    @staticmethod
    def format_number(x) -> str:
        return format(x, ",") if isinstance(x, int) else str(x)
