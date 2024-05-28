import inspect

from gba.client import ChatClient, Llama3Instruct
from gba.tools.base import Tool
from gba.utils import Scratchpad, exec_code, extract_code

SYSTEM_PROMPT = """Provide answers in Python code formatted as ```python <code> ```"""


USER_PROMPT_TEMPLATE = """The signature and documentation of the function to be called:

```
{function_spec}
```

Call the function for the following task:

```
{task}
```

Use the following additional context information if needed:

```
{context}
```

Generate a Python script that contains the function call arguments in a dict assigned to variable "arguments". Never add the function call itself."""


SPEC_TEMPLATE = '''def {name}{signature}:
    """{doc}"""
    ...'''


class FunctionCallTool(Tool):
    def __init__(self, model: Llama3Instruct, fn, summarizer=None):
        self.client = ChatClient(model=model)
        self.fn = fn
        self.fn_params = list(inspect.signature(fn).parameters.keys())
        self.summarizer = summarizer

    @property
    def name(self) -> str:
        return self.fn.__name__

    @property
    def doc(self) -> str:
        return self.fn.__doc__

    def run(self, request: str, task: str, scratchpad: Scratchpad, temperature: float = -1, **kwargs) -> str:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            function_spec=self._fn_spec(),
            context=scratchpad.results_repr(),
            task=task,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        message = self.client.complete(messages, temperature=temperature)
        code = extract_code(message["content"], remove_print_statements=True)

        arguments = exec_code(code, result_variable_name="arguments")
        arguments = {k: v for k, v in arguments.items() if k in self.fn_params}

        result = self.fn(**arguments)

        if self.summarizer is not None:
            return self.summarizer.summarize(task, result)
        else:
            return result

    def _fn_spec(self):
        return SPEC_TEMPLATE.format(name=self.name, signature=inspect.signature(self.fn), doc=self.fn.__doc__)
