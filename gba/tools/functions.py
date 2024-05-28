from datetime import date, time


def create_event(title: str, date: date, time: time | None = None) -> str:
    """Useful for creating an entry in a calendar."""
    return f"Event '{title}' successfully added to calendar, date={date}, time={time}"


def send_email(recipient: str, subject: str, body: str):
    """Useful for sending an email to a single recipient."""
    print(f"Email body: {body}")
    return f"Email sent to '{recipient}' with subject '{subject}'"


def search_images(query: str) -> str:
    """Useful for searching images matching a query."""
    name = query.replace(" ", "_")
    return f"[{name}_1.jpg](https://example.com/{name}_1.jpg)"
