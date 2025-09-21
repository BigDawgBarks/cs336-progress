from collections.abc import Iterable, Iterator
from tqdm import tqdm


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None=None):
        """Construct a tokenizer."""
        return None

    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: list[str] | None=None):
        """Construct and return Tokenizer from vocab and merges."""
        return None

    def encode(self, text: str) -> list[int]:
        """Encode input text into token IDs"""
        return []

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Returns generator lazily yielding token IDs, memory efficient."""
        yield -1

    def decode(self, ids: list[int]) -> str:
        """Decode token ids into text."""
        return ""
