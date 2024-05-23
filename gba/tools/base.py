from abc import ABC, abstractmethod
from typing import Dict, Iterable

from gba.utils import Scratchpad

TOOL_SPEC_TEMPLATE = """{name}: {doc}"""


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def doc(self) -> str | None:
        return self.__class__.run.__doc__

    @abstractmethod
    def run(self, request: str, task: str, scratchpad: Scratchpad, **kwargs) -> str: ...


class ToolsSpec(Dict[str, str]):
    def __init__(self, tools: Iterable[Tool]):
        super().__init__()
        for tool in tools:
            self[tool.name] = "" if tool.doc is None else tool.doc

    def sorted(self):
        spec = ToolsSpec([])
        for name, doc in sorted(self.items()):
            spec[name] = doc
        return spec

    def names_repr(self) -> str:
        return ", ".join(self.keys())

    def tools_repr(self) -> str:
        _tool_reprs = []
        for name, doc in self.items():
            _tool_repr = f"- {TOOL_SPEC_TEMPLATE.format(name=name, doc=doc)}"
            _tool_reprs.append(_tool_repr)
        return "\n".join(_tool_reprs)
