
from abc import ABC, abstractmethod


class BaseGenerator(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate an answer from a prompt.
        """
        pass
