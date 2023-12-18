from langchain.schema.messages import SystemMessage
from langchain_experimental.chat_models.llm_wrapper import ChatWrapper


class MistralInstruct(ChatWrapper):
    @property
    def _llm_type(self) -> str:
        return "mistral-instruct"

    sys_beg: str = "<s>[INST] "
    sys_end: str = "\n\n"
    ai_n_beg: str = "\n"
    ai_n_end: str = "</s>"
    usr_n_beg: str = "[INST] "
    usr_n_end: str = " [/INST]"
    usr_0_beg: str = ""
    usr_0_end: str = " [/INST]"
