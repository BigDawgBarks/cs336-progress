import base64
import regex
from collections.abc import Iterable, Iterator
from tqdm import tqdm


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None=None):
        """Construct a tokenizer."""
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        next_idx = max(list(vocab.keys())) + 1
        vocab_set = set(vocab.values())
        for token in self.special_tokens:
            if token not in vocab_set:
                vocab_set.add(token)
                self.vocab[next_idx] = token
                next_idx += 1


    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: list[str] | None=None):
        """Construct and return Tokenizer from vocab and merges."""
        with open(vocab_filepath, 'r') as vf:
            vocab_lines = vf.readlines()
        with open(merges_filepath, 'r') as mf:
            merge_lines = mf.readlines()

        # construct vocab {token}: {bytes}
        vocab = dict()
        for vocab_line in vocab_lines:
            parts = vocab_line.strip().split(': ')
            if len(parts) != 2:
                raise ValueError("Vocab is not formatted correctly. Each line"
                                 " must be {token}: {bytes}")
            id_str, bytes_str = parts
            token_id = int(id_str)
            token_bytes = base64.b64decode(bytes_str.encode('ascii'))
            vocab[token_id] = token_bytes

        # construct merges Merge {}, {}
        merges = []
        for merge_line in merge_lines:
            if merge_line[:6] != "Merge ":
                raise ValueError("Merges formatted incorrectly. Merge file line"
                                 " must begin with 'Merge: '")
            parts = merge_line[6:].strip().split(', ')
            if len(parts) != 2:
                raise ValueError("Merges formatted incorrectly. Each line must"
                                 " be 'Merge {token}, {token}'")
            token1, token2 = parts
            token1 = base64.b64decode(token1.encode('ascii'))
            token2 = base64.b64decode(token2.encode('ascii'))
            merges.append((token1, token2))

        return cls(vocab, merges, special_tokens)
                

    def encode(self, text: str) -> list[int]:
        """Encode input text into token IDs"""
        return list(self.encode_iterable(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Returns generator lazily yielding token IDs, memory efficient."""
        PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        CHUNK_SIZE = 100_000_000

        
        # loop:
        #   read chunk
        #   pretokenize chunk

        # end loop

        yield from ()

    def decode(self, ids: list[int]) -> str:
        """Decode token ids into text."""
        return b''.join([self.vocab[id_] for id_ in ids]).decode('utf-8')

if __name__ == "__main__":
    pass
