from pytest import fixture

from gba.agent import Scratchpad
from gba.tools.call.python import FunctionCallTool


def order_item(item: str, quantity: int) -> str:
    """Useful for ordering items in a store."""
    return f"Ordered {quantity} {item}."


@fixture(scope="module")
def call_tool(code_llama):
    yield FunctionCallTool(model=code_llama, fn=order_item)


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
