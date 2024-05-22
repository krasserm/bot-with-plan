import inspect

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.chat_models.llm_wrapper import ChatWrapper

from gba.tools.base import TOOL_DOC_TEMPLATE, Tool
from gba.utils import Scratchpad, exec_code, extract_code

SYSTEM_PROMPT = """Provide answers in Python wrapped into ```."""


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
    def __init__(self, model: ChatWrapper, fn, summarizer=None):
        self.model = model
        self.fn = fn
        self.fn_params = list(inspect.signature(fn).parameters.keys())
        self.summarizer = summarizer

    @property
    def name(self) -> str:
        return self.fn.__name__

    def doc(self):
        return TOOL_DOC_TEMPLATE.format(name=self.name, doc=self.fn.__doc__)

    def spec(self):
        return SPEC_TEMPLATE.format(name=self.name, signature=inspect.signature(self.fn), doc=self.fn.__doc__)

    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str:
        results = [se.result for se in scratchpad.entries]
        results_str = "\n".join(results)

        if not results_str:
            results_str = "<no context information available>"

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=USER_PROMPT_TEMPLATE.format(function_spec=self.spec(), context=results_str, task=task)
            ),
        ]

        ai_message = self.model.invoke(input=messages, prompt_ext=self.model.ai_n_beg)
        code = extract_code(ai_message.content.replace("```", "```python", 1))

        arguments = exec_code(code, result_variable_name="arguments")
        arguments = {k: v for k, v in arguments.items() if k in self.fn_params}

        result = self.fn(**arguments)

        if self.summarizer is not None:
            return self.summarizer.summarize(task, result)
        else:
            return result
