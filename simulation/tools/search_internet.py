import random

from gba.utils import Scratchpad
from simulation.tools.base import SimulatedTool

SYSTEM_PROMPT_ANSWER = """You pretend to be an internet search engine that retrieves information based on a task description and context information.
Formulate your answer as single sentence such that it is an answer to the task description. Pretend that you have access to real-time information.
If you don't know the answer, make a best guess. It is acceptable if the answer is incorrect.

Avoid referring the user to other sources.
Avoid including "as of" and your latest update in you answers."""


SYSTEM_PROMPT_PARTIAL_ANSWER = """You pretend to be an internet search engine that retrieves information based on a task description and context information.
You must only answer a part of the task description. Avoid providing a full answer.
If you cannot provide an incomplete answer, make a best guess. It is acceptable if the incomplete answer is incorrect.

Avoid including "as of" and your latest update in you answers."""


SYSTEM_PROMPT_NO_ANSWER = """You pretend to be an internet search engine that is unable to find the information requested in the task description and context information.
Your response should be a single sentence that indicates that you were unable to find the information requested.
Do not provide any additional information."""


USER_PROMPT_TEMPLATE = """Task description:

```
{task}
```

Context information:

```
{context}
```

Pretend searching the internet and provide a response."""


class SearchInternet(SimulatedTool):
    name: str = "search_internet"

    def run(
        self,
        request: str,
        task: str,
        scratchpad: Scratchpad,
        no_answer_prob: float = 0.1,
        partial_answer_prob: float = 0.1,
        **kwargs,
    ) -> str:
        """Useful for up-to-date information on the internet."""

        rand_num = random.random()
        if rand_num < no_answer_prob:
            system_prompt = SYSTEM_PROMPT_NO_ANSWER
        elif rand_num < no_answer_prob + partial_answer_prob:
            system_prompt = SYSTEM_PROMPT_PARTIAL_ANSWER
        else:
            system_prompt = SYSTEM_PROMPT_ANSWER

        user_prompt = USER_PROMPT_TEMPLATE.format(task=task, context=scratchpad.results_repr())
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.client.complete(messages)["content"]
