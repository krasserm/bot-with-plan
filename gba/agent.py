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
                tool_name = plan_result.get_selected_tool()
                tool_name = tool_name.replace("-", "_")

                if "," in tool_name:
                    tool_name = tool_name.split(",")[0]
                    tool_name = tool_name.strip()

                tool_name_orig = tool_name

                # deactivate the ask_user tool for automated evaluation of the agent
                if tool_name == "ask_user" and not self.ask_user:
                    tool_name = RespondTool.name

                if tool_name == "final_answer":
                    tool_name = RespondTool.name

                if tool_name == "calculate_number":
                    tool_name = CalculateTool.name

                if not tool_name:
                    tool_name = RespondTool.name

                if tool_name not in [RespondTool.name]:
                    print(f"Task: {task}")
                    print(f"Tool: {tool_name_orig}")

                tool = self.tools[tool_name]
                tool_result = tool.run(request, task, self.scratchpad)

                if tool_name != RespondTool.name:
                    print(f"Observation: {tool_result}")
                    print()

                self.scratchpad.add(task, tool_result)

                if tool_name == RespondTool.name:
                    if self.conversational:
                        self.history.append({"role": "user", "content": request})
                        self.history.append({"role": "assistant", "content": tool_result})
                    return tool_result

            return "I am unable to answer this request."
        finally:
            self.scratchpad.clear()
