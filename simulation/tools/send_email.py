from gba.utils import Scratchpad
from simulation.tools.base import SimulatedTool

SYSTEM_PROMPT = """You pretend to compose and send emails given a task description and context information.
You only need to answer that you've sent the email mentioning the subject and the recipient of the email. Avoid mentioning the email body.
If you cannot extract one of recipients, subject or the body of the email, you must answer that you cannot send the email and provide a reason.
Whatever your answer is, respond with a single sentence. This sentence must be as short as possible."""


USER_PROMPT = """Task description:

```
{task}
```

Context information:

```
{context}
```

Pretend sending and email and provide a response. Avoid responding in the first person"""


class SendEmail(SimulatedTool):
    name: str = "send_email"

    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str:
        """Useful for sending an email to a single recipient."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(task=task, context=scratchpad.results_repr())},
        ]
        return self.client.complete(messages)["content"]
