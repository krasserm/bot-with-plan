from gba.utils import Scratchpad
from simulation.tools.base import SimulatedTool


SYSTEM_PROMPT = """You are given a user request, context information and a task description. Provide an answer to the user request.
Use task description and context information as only source of information for answering the user request."""


USER_PROMPT_TEMPLATE = """User request:

```
{request}
```

Task description:

```
{task}
```

Context information:

```
{context}
```

Answer the user request in a single sentence."""


class FinalAnswer(SimulatedTool):
    name: str = "final_answer"

    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str:
        """Useful for providing the final answer to a request. Must always be used in the last step."""

        user_prompt = USER_PROMPT_TEMPLATE.format(request=request, task=task, context=scratchpad.results_repr())
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self.client.complete(messages)["content"]
