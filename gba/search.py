from gba.store import DocumentStore


class SearchEngine:
    def __init__(self, store: DocumentStore):
        self.store = store

    def search_internet(self, query: str) -> str:
        """Useful for searching factual and up-to-date information on the internet."""

        # -----------------------------------------------------------------------------------
        #  Internet search faked by searching through stored documents.
        # -----------------------------------------------------------------------------------

        documents, _ = self.store.search(query, n_results=2)
        return documents[0]

    @staticmethod
    def search_images(query: str) -> str:
        """Useful for searching images matching a query text."""

        # -----------------------------------------------------------------------------------
        #  Image search faked by returning query-dependent links.
        # -----------------------------------------------------------------------------------

        name = query.replace(" ", "_")
        return f"[{name}_1.jpg](https://example.com/{name}_1.jpg)"
