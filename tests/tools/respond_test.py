from pytest import fixture

from gba.tools import RespondTool
from gba.utils import Scratchpad, ScratchpadEntry


@fixture(scope="module")
def respond_tool(llama3_instruct):
    yield RespondTool(model=llama3_instruct)


def test_response_from_context(respond_tool):
    scratchpad = Scratchpad(
        entries=[
            ScratchpadEntry(task="Guess a name", result="Birgit"),
            ScratchpadEntry(task="Guess a name", result="Martin"),
        ]
    )

    response = respond_tool.run(request="Tell me two names", task="", scratchpad=scratchpad)

    assert "Birgit" in response
    assert "Martin" in response
