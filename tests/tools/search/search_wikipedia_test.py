import pytest

from gba.tools.search import SearchWikipediaTool, ContentExtractor
from gba.utils import Scratchpad


@pytest.fixture(scope="module")
def search_wikipedia_tool(llama3, embedding_model, rerank_model):
    yield SearchWikipediaTool(
        llm_model=llama3,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        top_k_nodes=10,
        top_k_related_documents=1,
        top_k_related_nodes=3,
        extractor=ContentExtractor(model=llama3),
    )


def test_search_wikipedia_without_related_info(search_wikipedia_tool, response_asserter):
    question = "Search Wikipedia for the launch date of the first iPhone."
    response = search_wikipedia_tool.run(
        task=question,
        request="",
        scratchpad=Scratchpad(),
    )

    response_asserter.assert_responses_similar(
        question=question,
        actual=response,
        expected="The launch date of the first iPhone was June 29, 2007.",
    )


def test_search_wikipedia_with_related_info(search_wikipedia_tool, response_asserter):
    question = "What are the physical characteristics of the Halicreas minimum?"
    response = search_wikipedia_tool.run(
        task=question,
        request="",
        scratchpad=Scratchpad(),
    )

    response_asserter.assert_responses_similar(
        question=question,
        actual=response,
        expected="The physical characteristics of Halicreas minimum include an umbrella 30–40 mm wide, thick, disk-like with a small apical projection, and 8 clusters of gelatinous papillae above the margin.",
    )


def test_search_wikipedia_with_unavailable_info(search_wikipedia_tool):
    response = search_wikipedia_tool.run(
        task="Who created the movie 'this is the bling' in 2024?",
        request="",
        scratchpad=Scratchpad(),
    )

    assert response.startswith("No information found")
