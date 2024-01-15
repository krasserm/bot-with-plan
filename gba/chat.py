from langchain_core.messages import SystemMessage
from langchain_experimental.chat_models.llm_wrapper import ChatWrapper


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
