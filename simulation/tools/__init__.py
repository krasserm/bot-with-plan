from typing import List

from gba.client import OpenAIClient
from simulation.tools.ask_user import AskUser
from simulation.tools.base import SimulatedTool
from simulation.tools.calculate_number import CalculateNumber
from simulation.tools.create_event import CreateEvent
from simulation.tools.final_answer import FinalAnswer
from simulation.tools.search_internet import SearchInternet
from simulation.tools.search_wikipedia import SearchWikipedia
from simulation.tools.send_email import SendEmail
from simulation.tools.use_bash import UseBash

TOOL_CLASSES = [
    AskUser,
    CalculateNumber,
    CreateEvent,
    SearchWikipedia,
    SearchInternet,
    SendEmail,
    UseBash,
    FinalAnswer,
]


def tools(client: OpenAIClient) -> List[SimulatedTool]:
    return [tool_class(client) for tool_class in TOOL_CLASSES]  # type: ignore
