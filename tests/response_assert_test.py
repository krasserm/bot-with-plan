import pytest

from tests.helpers.response_assert import ResponseAsserter


@pytest.fixture(scope="module")
def response_asserter(llama3):
    yield ResponseAsserter(llama3)


@pytest.mark.parametrize(
    "question, actual, expected, similar",
    [
        (
            "What is the capital of France?",
            "Paris is the capital of France.",
            "The capital city of France is Paris.",
            True,
        ),
        (
            "What does a dog sound like?",
            "A dog barks.",
            "Dogs make a barking sound.",
            True,
        ),
        (
            "What is the color of the sky on a clear day?",
            "The sky is blue on a clear day.",
            "On a clear day, the sky appears blue.",
            True,
        ),
        (
            "How do you make tea?",
            "Boil water, then steep the tea bag for a few minutes.",
            "Start by heating water to a boil, after which you immerse the tea bag and allow it to infuse.",
            True,
        ),
        (
            "What is the primary purpose of a refrigerator?",
            "A refrigerator is used to keep food cold and fresh.",
            "It’s designed to preserve perishable goods by maintaining a cool environment.",
            True,
        ),
        (
            "Why do we use sunscreen?",
            "Sunscreen protects our skin from harmful UV rays.",
            "To block the damaging effects of the sun’s ultraviolet radiation.",
            True,
        ),
        (
            "What is the capital of France?",
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            False,
        ),
        (
            "What does a dog sound like?",
            "A dog barks.",
            "Cats usually meow.",
            False,
        ),
        (
            "What is the color of the sky on a clear day?",
            "The sky is blue on a clear day.",
            "The grass is green.",
            False,
        ),
        (
            "How do you make tea?",
            "Boil water and add a tea bag.",
            "I prefer coffee, which involves brewing ground coffee beans.",
            False,
        ),
        (
            "What is the primary purpose of a refrigerator?",
            "A refrigerator is used to keep food cold.",
            "Freezers are great for long-term storage of meats and frozen goods.",
            False,
        ),
        (
            "Why do we use sunscreen?",
            "To protect our skin from UV rays.",
            "Moisturizers help keep the skin hydrated.",
            False,
        ),
    ],
)
def test_response_asserter(response_asserter, question, actual, expected, similar):
    assert response_asserter.responses_similar(question, actual, expected) == similar
