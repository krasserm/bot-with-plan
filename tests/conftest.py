from tempfile import TemporaryDirectory

import pytest
from sentence_transformers import SentenceTransformer, CrossEncoder

from gba.client import LlamaCppClient, MistralInstruct, Llama3Instruct
from tests.helpers.response_assert import ResponseAsserter


def pytest_addoption(parser):
    parser.addini("searxng_endpoint", "")
    parser.addini("mistral_instruct_endpoint", "")
    parser.addini("llama3_instruct_endpoint", "")


@pytest.fixture(scope="session")
def searxng_endpoint(request):
    yield request.config.getini("searxng_endpoint")


@pytest.fixture(scope="module")
def temp_dir():
    with TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="session")
def llama3_instruct(request):
    yield Llama3Instruct(llm=LlamaCppClient(url=request.config.getini("llama3_instruct_endpoint"), temperature=-1))


@pytest.fixture(scope="session")
def mistral_instruct(request):
    yield MistralInstruct(llm=LlamaCppClient(url=request.config.getini("mistral_instruct_endpoint"), temperature=-1))


@pytest.fixture(scope="session")
def embedding_model():
    yield SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cuda")


@pytest.fixture(scope="session")
def rerank_model():
    yield CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1", device="cuda")


@pytest.fixture(scope="session")
def response_asserter(llama3_instruct):
    yield ResponseAsserter(llama3_instruct)
