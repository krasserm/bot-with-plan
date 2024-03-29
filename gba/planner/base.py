from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from gba.utils import Scratchpad
from gba.client import Client, Message


class PlanResult(ABC):
    @abstractmethod
    def get_task(self) -> str:
        ...

    @abstractmethod
    def get_selected_tool(self) -> str:
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, str]:
        ...


class Planner(ABC):
    def __init__(self, client: Client):
        self.client = client

    @abstractmethod
    def plan(
            self,
            request: str,
            scratchpad: Scratchpad,
            history: Optional[List[Message]] = None,
            **kwargs,
    ) -> PlanResult:
        ...
