from abc import ABC, abstractmethod

class Tokenizer(ABC):
    
    @abstractmethod
    def encode(self, string: str) -> list[int]:
        pass
    
    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass