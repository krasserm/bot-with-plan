import json
from abc import ABC, abstractmethod
from typing import List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_experimental.chat_models.llm_wrapper import ChatWrapper
from pydantic import BaseModel

from gba.utils import prop_order_from_schema


TOOL_DOC_TEMPLATE = """{name}: {doc}"""


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


class PlanResult(BaseModel):
    context_information_summary: str
    thoughts: str
    task: str
    selected_action: str


class ScratchpadEntry(BaseModel):
    task: str
    result: str

    def __str__(self):
        return f"Task: {self.task}\nResult: {self.result}"


class Scratchpad(BaseModel):
    entries: list[ScratchpadEntry] = []

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def clear(self):
        self.entries = []

    def add(self, task: str, result: str):
        self.entries.append(ScratchpadEntry(task=task, result=result))

    def __str__(self):
        return "\n\n".join(str(entry) for entry in self.entries)


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str:
        ...

    def doc(self):
        return TOOL_DOC_TEMPLATE.format(name=self.name, doc=self.run.__doc__)


class Agent:
    def __init__(self, model: ChatWrapper, tools: List[Tool], conversational: bool = False):
        self.model = model
        self.tools = {tool.name: tool for tool in sorted(tools, key=lambda tool: tool.name)}
        self.conversational = conversational
        self.scratchpad = Scratchpad()
        self.history = []

    def plan(self, request: str) -> PlanResult:
        action_docs = "\n".join([f"- {tool.doc()}" for tool in self.tools.values()])
        action_names = ", ".join(self.tools.keys())

        if self.scratchpad.is_empty():
            context_str = "<no previous steps available>"
        else:
            context_str = str(self.scratchpad)

        user_prompt = PROMPT_TEMPLATE.format(
            action_docs=action_docs,
            action_names=action_names,
            request=request,
            context=context_str,
        )

        messages = self.history + [HumanMessage(content=user_prompt)]
        schema = PlanResult.model_json_schema()

        response = self.model.invoke(
            input=messages,
            schema=schema,
            prop_order=prop_order_from_schema(schema),
            prompt_ext=self.model.ai_n_beg,
        )

        return PlanResult(**json.loads(response.content))

    def run(self, request: str) -> str:
        from gba.tools.ask import AskTool
        from gba.tools.respond import RespondTool

        while True:
            plan_result = self.plan(request)

            task = plan_result.task
            action = plan_result.selected_action
            action = action.replace("-", "_")

            if "," in action:
                action = action.split(",")[0]
                action = action.strip()

            if not action:
                action = RespondTool.name

            if action not in [AskTool.name, RespondTool.name]:
                print(f"Task: {plan_result.task}")

            tool = self.tools[action]
            tool_result = tool.run(request, task, self.scratchpad)

            if action != RespondTool.name:
                print(f"Observation: {tool_result}")
                print()

            self.scratchpad.add(task, tool_result)

            if action == RespondTool.name:
                if self.conversational:
                    self.history.append(HumanMessage(content=request))
                    self.history.append(AIMessage(content=tool_result))
                self.scratchpad.clear()
                return tool_result
