from gba.utils import Scratchpad, extract_code
from simulation.tools.base import SimulatedTool


SYSTEM_PROMPT = """You pretend to execute commands in a Linux bash given a task description and context information.
You only need to provide the execution result as answer surrounded by triple backticks. Avoid mentioning which commands you have executed.
If the task description is not a usual task being executed in a bash, or if the task description and context doesn't provide enough information you must answer that you cannot execute the task and provide a reason in a short sentence."""


USER_PROMPT = """Task description:

```
{task}
```

Context information:

```
{context}
```

Pretend executing the task in a Linux bash and provide a response."""


class UseBash(SimulatedTool):
    name: str = "use_bash"

    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str:
        """Useful for executing commands in a Linux bash."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(task=task, context=scratchpad.results_repr())},
        ]
        message = self.client.complete(messages)
        return extract_code(message["content"]).strip()
