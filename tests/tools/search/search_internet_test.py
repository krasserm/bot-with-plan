import pytest

from gba.tools.search import ContentExtractor, SearchInternetTool
from gba.utils import Scratchpad


@pytest.fixture(scope="module")
def search_internet_tool(searxng_endpoint, llama3, rerank_model):
    yield SearchInternetTool(
        llm=llama3,
        rerank_model=rerank_model,
        searxng_endpoint=searxng_endpoint,
        fetch_webpage_timeout=5.0,
        top_k_documents=3,
        top_k_nodes_per_document=5,
        top_k_snippets=None,
        extractor=ContentExtractor(model=llama3),
    )


def test_search_internet(search_internet_tool, response_asserter):
    question = "Search for the launch date of the first iPhone"
    response = search_internet_tool.run(
        task=question,
        request="",
        scratchpad=Scratchpad(),
    )

    response_asserter.assert_responses_similar(
        question=question,
        actual=response,
        expected="The launch date of the first iPhone was June 29, 2007.",
    )


def test_search_internet_with_dates(search_internet_tool, response_asserter):
    question = "When was the video game 'The Last of Us' released"
    response = search_internet_tool.run(
        task=question,
        request="",
        scratchpad=Scratchpad(),
    )

    response_asserter.assert_responses_similar(
        question=question,
        actual=response,
        expected="The video game 'The Last of Us' was released for the PlayStation 3 in June 2013.",
    )


def test_search_internet_with_unavailable_info(search_internet_tool):
    response = search_internet_tool.run(
        task="Who created the movie 'this is the bling' in 2024?",
        request="",
        scratchpad=Scratchpad(),
    )

    assert response.startswith("No information found")
