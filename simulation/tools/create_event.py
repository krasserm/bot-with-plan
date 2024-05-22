from gba.utils import Scratchpad
from simulation.tools.base import SimulatedTool

SYSTEM_PROMPT = """You pretend to be a calendar supporting users to create an event given a task description and context information.
You must extract the event name, a concrete date value and and an optional time value from the task description and context information to create the event.
If any the event name or a concrete date values cannot be extracted, you must answer that you cannot create the event and provide a reason.

You must reject non-concrete date values like "today", "tomorrow", "next week" etc.
Avoid responding in the first person.
Avoid using newline characters in your answers."""


USER_PROMPT_TEMPLATE = """Task description:

```
{task}
```

Context information:

```
{context}
```

Pretend creating an event in a calendar and respond with a short sentence."""


class CreateEvent(SimulatedTool):
    name: str = "create_event"

    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str:
        """Useful for adding a single entry to my calendar at given date and time."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(task=task, context=scratchpad.results_repr())},
        ]
        return self.client.complete(messages)["content"]
