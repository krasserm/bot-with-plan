from tempfile import TemporaryDirectory

import pytest
from langchain_experimental.chat_models.llm_wrapper import Llama2Chat
from sentence_transformers import SentenceTransformer, CrossEncoder

from gba.client import LlamaCppClient, MistralInstruct, Llama3Instruct
from tests.helpers.response_assert import ResponseAsserter


def pytest_addoption(parser):
    parser.addini("searxng_endpoint", "")


@pytest.fixture(scope="session")
def searxng_endpoint(request):
    yield request.config.getini("searxng_endpoint")


@pytest.fixture(scope="module")
def temp_dir():
    with TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="session")
def mistral_instruct():
    yield MistralInstruct(llm=LlamaCppClient(url="http://localhost:8081/completion", temperature=-1))


@pytest.fixture(scope="session")
def code_llama():
    yield Llama2Chat(llm=LlamaCppClient(url="http://localhost:8088/completion", temperature=-1))


@pytest.fixture(scope="session")
def llama3():
    yield Llama3Instruct(llm=LlamaCppClient(url="http://localhost:8084/completion", temperature=-1))


@pytest.fixture(scope="session")
def embedding_model():
    yield SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cuda")


@pytest.fixture(scope="session")
def rerank_model():
    yield CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1", device="cuda")


@pytest.fixture(scope="session")
def response_asserter(llama3):
    yield ResponseAsserter(llama3)
