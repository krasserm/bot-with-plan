from tempfile import TemporaryDirectory

import pytest
from langchain_experimental.chat_models.llm_wrapper import Llama2Chat

from gba.client import LlamaCppClient, MistralInstruct


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
