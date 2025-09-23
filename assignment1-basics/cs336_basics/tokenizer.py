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
        CHUNK_SIZE = 1_000_000

        compiled_pattern = regex.compile(PRETOKENIZATION_PATTERN)
        accumulated_text = ""
        prefix = ""  # Prefix from previous chunk

        bytes_to_token_id = {v: k for k, v in self.vocab.items()}

        # Persistent mapping of pretoken/special_token -> token_ids for efficiency
        token_cache = {}
        for text_chunk in iterable:
            accumulated_text += text_chunk

            while len(accumulated_text) >= CHUNK_SIZE:
                chunk_text = prefix + accumulated_text[:CHUNK_SIZE]
                accumulated_text = accumulated_text[CHUNK_SIZE:]

                # First pass: split on special tokens
                if self.special_tokens:
                    special_pattern = f"({'|'.join(regex.escape(token) for token in self.special_tokens)})"
                    parts = regex.split(special_pattern, chunk_text)
                    parts = [part for part in parts if part]  # Remove empty parts
                else:
                    parts = [chunk_text]

                # Second pass: pretokenize non-special parts
                processed_parts = []
                for part in parts:
                    if part in self.special_tokens:
                        processed_parts.append(part)
                    else:
                        pretokens = [match.group(0) for match in compiled_pattern.finditer(part)]
                        processed_parts.extend(pretokens)

                if processed_parts:
                    prefix = processed_parts.pop()
                else:
                    prefix = ""

                for token in processed_parts:
                    if token not in token_cache:
                        if token in self.special_tokens:
                            token_bytes = token.encode('utf-8')
                            if token_bytes in bytes_to_token_id:
                                token_cache[token] = [bytes_to_token_id[token_bytes]]
                            else:
                                token_cache[token] = []
                        else:
                            token_cache[token] = self._apply_bpe_to_token(token, bytes_to_token_id)

                for token in processed_parts:
                    yield from token_cache[token]

        if accumulated_text or prefix:
            final_text = prefix + accumulated_text

            if self.special_tokens:
                special_pattern = f"({'|'.join(regex.escape(token) for token in self.special_tokens)})"
                parts = regex.split(special_pattern, final_text)
                parts = [part for part in parts if part]  # Remove empty parts
            else:
                parts = [final_text]

            processed_parts = []
            for part in parts:
                if part in self.special_tokens:
                    processed_parts.append(part)
                else:
                    pretokens = [match.group(0) for match in compiled_pattern.finditer(part)]
                    processed_parts.extend(pretokens)

            for token in processed_parts:
                if token not in token_cache:
                    if token in self.special_tokens:
                        token_bytes = token.encode('utf-8')
                        if token_bytes in bytes_to_token_id:
                            token_cache[token] = [bytes_to_token_id[token_bytes]]
                        else:
                            token_cache[token] = []
                    else:
                        token_cache[token] = self._apply_bpe_to_token(token, bytes_to_token_id)

            for token in processed_parts:
                yield from token_cache[token]

    def _apply_bpe_to_token(self, token: str, bytes_to_token_id: dict) -> list[int]:
        """Apply BPE merges to a single token and return list of token IDs."""
        token_bytes = token.encode('utf-8')
        symbols = [bytes([b]) for b in token_bytes]

        for merge_pair in self.merges:
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols) - 1 and
                    symbols[i] == merge_pair[0] and
                    symbols[i + 1] == merge_pair[1]):
                    merged = merge_pair[0] + merge_pair[1]
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        token_ids = []
        for symbol in symbols:
            if symbol in bytes_to_token_id:
                token_ids.append(bytes_to_token_id[symbol])
            else:
                # Fallback to individual bytes if symbol not found
                for b in symbol:
                    token_ids.append(bytes_to_token_id.get(bytes([b]), 0))

        return token_ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids into text."""
        byte_sequence = b''.join([self.vocab[id_] for id_ in ids])
        try:
            return byte_sequence.decode('utf-8')
        except UnicodeDecodeError:
            if len(ids) == 1:
                return ""
            else:
                # For multiple tokens, this shouldn't happen in normal cases
                # Re-raise the original error
                raise

if __name__ == "__main__":
    pass

