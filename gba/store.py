import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder


class DocumentStore:
    def __init__(self, path=".chroma"):
        self.sentence_embedder = embedding_functions.SentenceTransformerEmbeddingFunction()
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name="functions",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.sentence_embedder,
        )

    def add(self, identifier: str, document: str):
        self.collection.add(ids=identifier, documents=document)

    def count(self):
        return self.collection.count()

    def search(self, query: str, n_results=1):
        results = self.collection.query(query_texts=[query], n_results=n_results)
        documents = results["documents"][0]

        pairs = []

        for doc in documents:
            pairs.append([query, doc])

        scores = self.cross_encoder.predict(pairs)
        documents = np.array(documents, dtype=object)

        order = np.argsort(scores)[::-1]
        return documents[order].tolist(), scores[order].tolist()
