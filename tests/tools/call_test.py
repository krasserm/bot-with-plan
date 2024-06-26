from pytest import fixture

from gba.tools.call import FunctionCallTool
from gba.utils import Scratchpad


def order_item(item: str, quantity: int) -> str:
    """Useful for ordering items in a store."""
    return f"Ordered {quantity} {item}."


@fixture(scope="module")
def call_tool(llama3_instruct):
    yield FunctionCallTool(model=llama3_instruct, fn=order_item)


def test_call_with_empty_scratchpad(call_tool):
    response = call_tool.run(
        request="",
        task="Order 3 apples",
        scratchpad=Scratchpad(),
    )
    assert "3 apple" in response.lower()


def test_call(call_tool):
    scratchpad = Scratchpad()
    scratchpad.add(
        task="Find out what is available in the store.",
        result="Only apples are available.",
    )

    response = call_tool.run(
        request="",
        task="Order 3 of them",
        scratchpad=scratchpad,
    )
    assert "3 apple" in response.lower()
