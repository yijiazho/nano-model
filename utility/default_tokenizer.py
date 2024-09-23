import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utility.tokenizer import Tokenizer

class DefaultTokenizer(Tokenizer):
    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        self.stoi = { ch:i for i, ch in enumerate(self.chars) }
        self.itos = { i:ch for i, ch in enumerate(self.chars) }
    
    def vocab_size(self) -> int:
        return len(self.chars)
    
    def encode(self, string: str) -> list[int]:
        return [self.stoi[c] for c in string]
    
    def decode(self, tokens: list[int]) -> str:
        return ''.join([self.itos[i] for i in tokens])
    