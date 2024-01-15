import pytest

from gba.store import DocumentStore


@pytest.fixture(scope="module")
def store(temp_dir):
    store = DocumentStore(path=temp_dir)
    store.add(identifier="1", document="Document 1 is about cats.")
    store.add(identifier="2", document="Document 2 is about dogs.")
    yield store


def test_store(store):
    documents, scores = store.search(query="dogs", n_results=2)
    assert documents == [
        "Document 2 is about dogs.",
        "Document 1 is about cats.",
    ]
    assert scores[0] > scores[1]
