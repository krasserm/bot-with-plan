import inspect

from langchain_core.language_models import LLM

from gba.tools.base import Tool, TOOL_DOC_TEMPLATE
from gba.utils import parse_function_call, Scratchpad


PROMPT_TEMPLATE = '''
Function:
{function_spec}

User query: {task}

Use the following additional context information if needed:  

```
{context}
```<human_end>

'''


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

    def doc(self):
        return TOOL_DOC_TEMPLATE.format(name=self.name, doc=self.fn.__doc__)

    def spec(self):
        return SPEC_TEMPLATE.format(name=self.name, signature=inspect.signature(self.fn), doc=self.fn.__doc__)

    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str:
        results = [se.result for se in scratchpad.entries]
        results_str = "\n".join(results)

        if not results_str:
            results_str = "<no context information available>"

        prompt = PROMPT_TEMPLATE.format(
            function_spec=self.spec(),
            context=results_str,
            task=task
        )

        fn_response = self.model.invoke(prompt, stop=["<bot_end>"])
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
