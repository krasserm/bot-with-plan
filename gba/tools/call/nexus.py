import inspect

from langchain_core.language_models import LLM

from gba.tools.base import Tool
from gba.utils import Scratchpad, parse_function_call

PROMPT_TEMPLATE = """
Function:
{function_spec}

User query: {task}

Use the following additional context information if needed:

```
{context}
```<human_end>

"""


SPEC_TEMPLATE = '''def {name}{signature}:
    """{doc}"""'''


class FunctionCallTool(Tool):
    def __init__(self, model: LLM, fn, summarizer=None):
        self.model = model
        self.fn = fn
        self.summarizer = summarizer

    @property
    def name(self) -> str:
        return self.fn.__name__

    @property
    def doc(self) -> str:
        return self.fn.__doc__

    def run(self, request: str, task: str, scratchpad: Scratchpad, temperature: float = -1, **kwargs) -> str:
        prompt = PROMPT_TEMPLATE.format(function_spec=self._fn_spec(), context=scratchpad.results_repr(), task=task)

        fn_response = self.model.invoke(prompt, stop=["<bot_end>"], temperature=temperature)
        fn_call = fn_response[6:]

        # ------------------------------------------------
        #  TODO: also support non-builtin parameter types
        # ------------------------------------------------
        fn_args, fn_kwargs = parse_function_call(fn_call)
        result = self.fn(*fn_args, **fn_kwargs)

        print(f"Call: {fn_call}")

        if self.summarizer is not None:
            return self.summarizer.summarize(task, result)
        else:
            return result

    def _fn_spec(self):
        return SPEC_TEMPLATE.format(name=self.name, signature=inspect.signature(self.fn), doc=self.fn.__doc__)
