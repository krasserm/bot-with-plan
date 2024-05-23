import json
from typing import List, Tuple

from gba.planner import Planner, PlanResult
from gba.utils import Scratchpad, ScratchpadEntry
from simulation.tools import FinalAnswer, SearchInternet, SearchWikipedia, SimulatedTool


class Agent:
    def __init__(self, planner: Planner, tools: List[SimulatedTool]):
        self.planner = planner
        self.tools = {tool.name: tool for tool in tools}

    def run(
        self,
        request: str,
        max_steps: int = 10,
        verbose: bool = True,
        **planner_kwargs,
    ) -> Tuple[List[PlanResult], List[ScratchpadEntry]]:
        scratchpad = Scratchpad()
        plans = []
        tool = ""

        for _ in range(max_steps):
            plan_result = self.planner.plan(request, scratchpad, **planner_kwargs)
            plans.append(plan_result)

            if verbose:
                print(json.dumps(plan_result.to_dict(), indent=2))

            task = plan_result.get_task()
            tool = plan_result.get_selected_tool()

            if tool not in self.tools:
                raise ValueError(f"Hallucinated  tool '{tool}'")

            tool_kwargs = {}
            if tool in [SearchInternet.name, SearchWikipedia.name]:
                tool_kwargs["no_answer_prob"] = 0.1
                tool_kwargs["partial_answer_prob"] = 0.1

            tool_obj = self.tools[tool]
            tool_res = tool_obj.run(request=request, task=task, scratchpad=scratchpad, **tool_kwargs)

            if verbose:
                print(f"{tool} result: {tool_res}")

            scratchpad.add(task=task, result=tool_res)

            if tool == FinalAnswer.name:
                break

        assert len(plans) == len(scratchpad.entries)

        if tool == "final_answer":
            return plans, scratchpad.entries
        else:
            return [], []
