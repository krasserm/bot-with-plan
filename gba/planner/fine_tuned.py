import json
from typing import Dict, List, Optional

from pydantic import BaseModel

from gba.client import Message
from gba.planner import Planner, PlanResult
from gba.utils import Scratchpad

PROMPT_TEMPLATE = """User request:

```
{request}
```

Context information:

```
{context}
```

Plan the next step."""


class _PlanResult(BaseModel, PlanResult):
    context_information_summary: str
    thoughts: str
    task: str
    selected_tool: str

    def get_task(self) -> str:
        return self.task

    def get_selected_tool(self) -> str:
        return self.selected_tool

    def to_dict(self) -> Dict[str, str]:
        return self.dict()


class FineTunedPlanner(Planner):
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

    @staticmethod
    def create_messages(request: str, scratchpad: Scratchpad) -> List[Message]:
        prompt = PROMPT_TEMPLATE.format(request=request, context=scratchpad.entries_repr())
        return [{"role": "user", "content": prompt}]
