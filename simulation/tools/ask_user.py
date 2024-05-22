from gba.utils import Scratchpad
from simulation.tools.base import SimulatedTool

SYSTEM_PROMPT = """You are a user interacting with an agent. The agent sometimes asks questions which you have to answer.
Your answer should be as short as possible. If possible use keywords or short phrases to answer the questions. Otherwise use a short sentence.
When asked for personal information, provide fictional data. Avoid using newline characters in your answers."""


USER_PROMPT_TEMPLATE = """Here is the question from the AI agent:

```
{question}
```

Provide your answer."""


class AskUser(SimulatedTool):
    name: str = "ask_user"

    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str:
        """Useful for asking user about information missing in the request."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=task)},
        ]
        return self.client.complete(messages)["content"]
