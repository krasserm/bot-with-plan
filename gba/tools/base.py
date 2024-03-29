from abc import ABC, abstractmethod

from gba.utils import Scratchpad


TOOL_DOC_TEMPLATE = """{name}: {doc}"""


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str:
        ...

    @classmethod
    def doc(cls) -> str:
        return TOOL_DOC_TEMPLATE.format(name=cls.name, doc=cls.run.__doc__)
