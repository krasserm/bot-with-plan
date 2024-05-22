from abc import ABC, abstractmethod
from typing import Dict, List

Message = Dict[str, str]


class Client(ABC):
    @abstractmethod
    def complete(self, messages: List[Message], **kwargs) -> Message: ...
