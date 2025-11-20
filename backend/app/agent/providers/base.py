from abc import ABC, abstractmethod
from typing import Any

class Provider(ABC):
    """
    Abstract base class for all agent providers.
    """

    @abstractmethod
    async def __call__(self,prompt: str, **generation_args: Any) -> str: ...

class EmbeddingProvider(ABC):
    """
    Abstract base class for all embedding providers.
    """

    @abstractmethod
    async def embed(self,text: str) -> list[float]: ...
  