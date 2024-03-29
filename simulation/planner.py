import json
from typing import Dict

from pydantic import BaseModel

from gba.client import OpenAIClient
from gba.planner import Planner, PlanResult
from gba.utils import Scratchpad
from simulation.tools import tools_string


SYSTEM_PROMPT = """You are given a user request and context information. You can select one of the following tools:

```
{tools}
```

Always answer in the following JSON format:

{{
  "user_request": <the user request>,
  "context_information_summary": <summary of available context information>,
  "missing_information_or_action": <information or action that is still missing to fully answer the user request>,
  "thoughts": <your thoughts about what to do in a next step>,
  "task": <task description for the next step in natural language, including relevant input values>,
  "selected_tool": <the name of the selected tool for the next step. Use one of the available tools>,
}}

For selecting the next tool or providing a final answer, always use context information only.

The generated task should be a single task. 
Avoid combining multiple tasks in one step. If you need to perform multiple tasks, do them one after the other.
Avoid requesting more than one piece of information in a single task. If you need multiple pieces of information request them one after the other."""


USER_PROMPT = """User request:

```
{request}
```

Context information:

```
{context}
```

{instruction}"""


class OpenAIPlanResult(BaseModel, PlanResult):
    user_request: str
    context_information_summary: str
    missing_information_or_action: str
    thoughts: str
    task: str
    selected_tool: str

    def get_task(self) -> str:
        return self.task

    def get_selected_tool(self) -> str:
        return self.selected_tool

    def to_dict(self) -> Dict[str, str]:
        return self.dict()


class OpenAIPlanner(Planner):
    def __init__(self, client: OpenAIClient):
        super().__init__(client)

    def plan(
            self,
            request: str,
            scratchpad: Scratchpad,
            temperature: float = 0.1,
            direct_answer: bool = False,
            **kwargs,
    ) -> OpenAIPlanResult:
        if direct_answer:
            instruction = ("Provide a final answer to the user request. "
                           "You must only use the tool final_answer, no other tool.")
        else:
            instruction = "Provide the next step to answer the user request."

        system_prompt = SYSTEM_PROMPT.format(tools=tools_string())
        user_prompt = USER_PROMPT.format(request=request, context=scratchpad.entries_repr(), instruction=instruction)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        message = self.client.complete(messages, enforce_json_output=True, temperature=temperature)
        message_json = json.loads(message["content"])
        return OpenAIPlanResult(**message_json)
