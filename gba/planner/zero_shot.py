import json
from typing import Dict, List, Optional

from pydantic import BaseModel

from gba.client import ChatClient, Message
from gba.planner import Planner, PlanResult
from gba.tools.base import ToolsSpec
from gba.utils import Scratchpad


PROMPT_TEMPLATE = """You are given a user request and context information. You can select one of the following actions:

```
{action_docs}
```

User request:

```
{request}
```

Context information:

```
{context}
```

Always answer in the following JSON format:

{{
  "context_information_summary": <summary of available context information. Always summarize calculation results if present>,
  "thoughts": <your thoughts about what to do in a next step>,
  "task": <task description for the next step in natural language, including relevant input values>,
  "selected_action": <the name of the selected action for the next step. Use one of [{action_names}]>
}}

For selecting the next action or providing a final answer, always use context information only."""


class _PlanResult(BaseModel, PlanResult):
    context_information_summary: str
    thoughts: str
    task: str
    selected_action: str

    def get_task(self) -> str:
        return self.task

    def get_selected_tool(self) -> str:
        return self.selected_action

    def to_dict(self) -> Dict[str, str]:
        plan_dict = self.dict()
        plan_dict["selected_tool"] = plan_dict.pop("selected_action")
        return plan_dict


class ZeroShotPlanner(Planner):
    def __init__(self, client: ChatClient, tools_spec: ToolsSpec):
        super().__init__(client)
        self.tools_spec = tools_spec.sorted()

    def plan(
        self,
        request: str,
        scratchpad: Scratchpad,
        history: Optional[List[Message]] = None,
        temperature: float = -1,
        **kwargs,
    ) -> PlanResult:
        messages = self.create_messages(request=request, scratchpad=scratchpad)

        if history is not None:
            messages = history + messages

        message = self.client.complete(
            messages=messages,
            schema=_PlanResult.model_json_schema(),
            temperature=temperature,
        )
        message_json = json.loads(message["content"])
        return _PlanResult(**message_json)

    def create_messages(self, request: str, scratchpad: Scratchpad) -> List[Message]:
        prompt = PROMPT_TEMPLATE.format(
            action_docs=self.tools_spec.tools_repr(),
            action_names=self.tools_spec.names_repr(),
            request=request,
            context=scratchpad.entries_repr(),
        )

        return [{"role": "user", "content": prompt}]
