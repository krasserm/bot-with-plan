from abc import ABC

from gba.client import OpenAIClient
from gba.tools.base import Tool


class SimulatedTool(Tool, ABC):
    def __init__(self, client: OpenAIClient):
        self.client = client
