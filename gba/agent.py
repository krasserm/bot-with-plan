from typing import List

from gba.client import Message
from gba.planner import Planner
from gba.tools.base import Tool
from gba.utils import Scratchpad


class Agent:
    def __init__(self, planner: Planner, tools: List[Tool], conversational: bool = False, ask_user: bool = True):
        self.planner = planner
        self.tools = {tool.name: tool for tool in tools}
        self.conversational = conversational
        self.scratchpad = Scratchpad()
        self.history: List[Message] = []
        self.ask_user = ask_user

    def run(self, request: str, max_steps: int = 10) -> str:
        from gba.tools.ask import AskTool
        from gba.tools.calculate import CalculateTool
        from gba.tools.respond import RespondTool

        try:
            for _ in range(max_steps):
                plan_result = self.planner.plan(
                    request=request,
                    scratchpad=self.scratchpad,
                    history=self.history,
                )

                task = plan_result.get_task()
                action = plan_result.get_selected_tool()
                action = action.replace("-", "_")

                if "," in action:
                    action = action.split(",")[0]
                    action = action.strip()

                # deactivate the ask_user tool for automated evaluation of the agent
                if action == "ask_user" and not self.ask_user:
                    action = RespondTool.name

                if action == "final_answer":
                    action = RespondTool.name

                if action == "calculate_number":
                    action = CalculateTool.name

                if not action:
                    action = RespondTool.name

                if action not in [AskTool.name, RespondTool.name]:
                    print(f"Task: {task}")

                tool = self.tools[action]
                tool_result = tool.run(request, task, self.scratchpad)

                if action != RespondTool.name:
                    print(f"Observation: {tool_result}")
                    print()

                self.scratchpad.add(task, tool_result)

                if action == RespondTool.name:
                    if self.conversational:
                        self.history.append({"role": "user", "content": request})
                        self.history.append({"role": "assistant", "content": tool_result})
                    return tool_result

            return "I am unable to answer this request."
        finally:
            self.scratchpad.clear()
