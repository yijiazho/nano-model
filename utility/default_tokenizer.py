import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utility.tokenizer import Tokenizer

class DefaultTokenizer(Tokenizer):
    def __init__(self, path: str) -> None:
        with open(path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.chars = sorted(list(set(self.text)))
        self.stoi = { ch:i for i, ch in enumerate(self.chars) }
        self.itos = { i:ch for i, ch in enumerate(self.chars) }
    
    def encode(self, string: str) -> list[int]:
        return [self.stoi[c] for c in string]
    
    def decode(self, tokens: list[int]) -> str:
        return ''.join([self.itos[i] for i in tokens])
    