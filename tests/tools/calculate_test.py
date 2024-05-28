from pytest import approx, fixture

from gba.tools import CalculateTool
from gba.utils import Scratchpad


@fixture(scope="module")
def calculate_tool(llama3_instruct):
    yield CalculateTool(model=llama3_instruct)


def test_calculate(calculate_tool):
    scratchpad = Scratchpad()
    scratchpad.add(
        task="Search the internet for Martin's age.",
        result="Martin is 32 years old.",
    )

    response = calculate_tool.run(
        request="",
        task="What is Martin's age divided by 7?",
        scratchpad=scratchpad,
    )

    assert float(response) == approx(4.57143)
