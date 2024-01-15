from gba.agent import Scratchpad, Tool


class AskTool(Tool):
    name: str = "ask_user"

    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str:
        """Useful when you need additional information from the user."""
        return input(f"{task}: ")
