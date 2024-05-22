from gba.utils import Scratchpad
from simulation.tools.base import SimulatedTool

SYSTEM_PROMPT = """You pretend to be a calculator that outputs a single number given a task description and context information. This number may also be a date or time
If information is missing the the task description which you could also obtain from a Python interpreter, pretend to do so. Avoid mentioning which calculation steps you have made.
If the task description is not a numerical task that can be usually solved with a calculator you must answer that you cannot execute the task and provide a reason in a short sentence."""


USER_PROMPT_TEMPLATE = """Task description:

```
{task}
```

Context information:

```
{context}
```

Pretend doing the calculation and provide the result as a response."""


class CalculateNumber(SimulatedTool):
    name: str = "calculate_number"

    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str:
        """Useful for numerical tasks that result in a single number."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(task=task, context=scratchpad.results_repr())},
        ]
        return self.client.complete(messages)["content"]
