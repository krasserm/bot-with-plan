import numexpr as ne


# ------------------------------------
#  Tool mockups used in example_agent.ipynb
# ------------------------------------


def search_internet(query: str):
    """A tool for searching factual and up-to-date information matching 'query' text."""

    if "dicaprio" in query.lower() and "girlfriend" in query.lower():
        return "Leonardo di Caprio started dating Vittoria Ceretti in 2023. She was born in Italy and is 25 years old"


def search_images(query: str):
    """A tool for searching images with a given query text."""
    name = query.replace(" ", "_")
    return f"[{name}_1.jpg](https://example.com/{name}_1.jpg)"


def create_event(title: str, datetime: str):
    """A tool for adding an event to a calendar."""
    return f"Event {title} successfully created!"
