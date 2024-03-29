from pytest import fixture

from gba.utils import ScratchpadEntry, Scratchpad
from gba.tools import RespondTool


@fixture(scope="module")
def respond_tool(mistral_instruct):
    yield RespondTool(model=mistral_instruct)


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


def test_response_from_task(respond_tool):
    response = respond_tool.run(
        request="Tell me two names",
        task="Mention the names Birgit and Martin",
        scratchpad=Scratchpad(),
    )

    assert "Birgit" in response
    assert "Martin" in response
