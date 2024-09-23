import tiktoken
from utility.tokenizer import Tokenizer

class TiktokenTokenizer(Tokenizer):
    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        self.tokenizer = tiktoken.encoding_for_model(model)
    
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab
    
    def encode(self, string: str) -> list[int]:
        return self.tokenizer.encode(string)
    
    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)