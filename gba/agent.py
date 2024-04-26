from typing import List

from gba.planner import Planner
from gba.tools.base import Tool
from gba.utils import Scratchpad


class Agent:
    def __init__(self, planner: Planner, tools: List[Tool], conversational: bool = False):
        self.planner = planner
        self.tools = {tool.name: tool for tool in tools}
        self.conversational = conversational
        self.scratchpad = Scratchpad()
        self.history = []

    def run(self, request: str) -> str:
        from gba.tools.ask import AskTool
        from gba.tools.calculate import CalculateTool
        from gba.tools.respond import RespondTool

        while True:
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

            if action == "final_answer":
                action = RespondTool.name

            if action == "calculate_number":
                action = CalculateTool.name

            if action == "search_wikipedia":
                action = "search_internet"

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
                self.scratchpad.clear()
                return tool_result
