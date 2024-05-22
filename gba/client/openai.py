from typing import List, Optional

from openai import OpenAI

from gba.client.base import Client, Message


class OpenAIClient(Client):
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key, organization=organization)
        self.model = model

    def complete(self, messages: List[Message], enforce_json_output: bool = False, **kwargs) -> Message:
        if enforce_json_output:
            extras = {"response_format": {"type": "json_object"}}
        else:
            extras = {}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **extras,
            **kwargs,
        )
        return {"role": "assistant", "content": response.choices[0].message.content}
