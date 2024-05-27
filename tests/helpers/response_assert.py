import json

from gba.client import Llama3Instruct, ChatClient

ASSERT_SYSTEM_PROMPT = "You are a helpful assistant that validates if two answers are conceptually similar."


ASSERT_USER_PROMPT_TEMPLATE = """You are given a question, a reference response and an actual response. Evaluate the similarity of the responses based on their underlying concepts and context, not just their wording.
Your task is to analyze the core ideas and intents of each response to determine if they convey the same or similar meanings in answer to the given question.

Question: '{question}'
Reference Answer: '{expected}'
Actual Answer: '{actual}'

Are the responses conceptually similar or dissimilar? Only output the final result. Use the following output format:

{{
  "similar": <true / false>
}}
"""


class ResponseAsserter:
    def __init__(self, llm: Llama3Instruct):
        self._llm = llm
        self._client = ChatClient(llm)

    def responses_similar(self, question: str, actual: str, expected: str) -> bool:
        messages = [
            {"role": "system", "content": ASSERT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ASSERT_USER_PROMPT_TEMPLATE.format(question=question, expected=expected, actual=actual),
            },
        ]

        response = self._client.complete(messages, temperature=-1)
        text_response = response["content"].strip()

        try:
            json_response = json.loads(text_response)
            if "similar" in json_response:
                return json_response["similar"]
            else:
                raise ValueError(f"Unexpected response: {json_response}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {text_response}") from e

    def assert_responses_similar(self, question: str, actual: str, expected: str):
        assert self.responses_similar(
            question, actual, expected
        ), f"Responses are not similar. expected: {expected}, actual: {actual}"
