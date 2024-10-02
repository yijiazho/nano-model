from transformers import GPT2Tokenizer

from utility.tokenizer import Tokenizer

class MyGPT2Tokenizer(Tokenizer):
    def __init__(self) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size
    
    def encode(self, string: str, max_length: int = 1024) -> list[int]:
        return self.tokenizer.encode(string, max_length=max, truncation=True)
    
    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)
