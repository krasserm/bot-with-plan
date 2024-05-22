from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_experimental.chat_models.llm_wrapper import ChatWrapper

from gba.client.base import Client, Message
from gba.utils import prop_order_from_schema


class Llama3Instruct(ChatWrapper):
    @property
    def _llm_type(self) -> str:
        return "llama-3-instruct"

    sys_beg: str = "<|start_header_id|>system<|end_header_id|>\n\n"
    sys_end: str = "<|eot_id|>"
    ai_n_beg: str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ai_n_end: str = "<|eot_id|>"
    usr_n_beg: str = "<|start_header_id|>user<|end_header_id|>\n\n"
    usr_n_end: str = "<|eot_id|>"


class MistralInstruct(ChatWrapper):
    @property
    def _llm_type(self) -> str:
        return "mistral-instruct"

    sys_beg: str = ""
    sys_end: str = ""
    ai_n_beg: str = ""
    ai_n_end: str = "</s>"
    usr_n_beg: str = "[INST] "
    usr_n_end: str = " [/INST]"
    usr_0_beg: str = "[INST] "
    usr_0_end: str = " [/INST]"

    system_message = SystemMessage(content="")


class ChatML(ChatWrapper):
    @property
    def _llm_type(self) -> str:
        return "chat-ml"

    sys_beg: str = "<|im_start|>system\n"
    sys_end: str = "<|im_end|>\n"
    ai_n_beg: str = "<|im_start|>assistant\n"
    ai_n_end: str = "<|im_end|>\n"
    usr_n_beg: str = "<|im_start|>user\n"
    usr_n_end: str = "<|im_end|>\n"


class ChatClient(Client):
    def __init__(self, model: ChatWrapper):
        self.model = model

    @staticmethod
    def convert_messages(messages: List[Message]):
        messages_impl = []

        for message in messages:
            role = message["role"]
            if role == "system":
                messages_impl.append(SystemMessage(content=message["content"]))
            elif role == "user":
                messages_impl.append(HumanMessage(content=message["content"]))
            elif role == "assistant":
                messages_impl.append(AIMessage(content=message["content"]))
            else:
                raise ValueError(f"Unsupported role: {role}")

        return messages_impl

    def complete(self, messages: List[Message], schema=None, **kwargs) -> Message:
        if schema is None:
            schema_args = {}
        else:
            schema_args = {
                "schema": schema,
                "prop_order": prop_order_from_schema(schema),
            }

        response = self.model.invoke(
            input=self.convert_messages(messages),
            prompt_ext=self.model.ai_n_beg,
            **schema_args,
            **kwargs,
        )

        return {"role": "assistant", "content": response.content}
