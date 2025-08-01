"""Logic for Training BPE Tokenizer."""

from collections import defaultdict, Counter
from functools import partial
from cs336_basics.my_profiler import profile_block
from cs336_basics.pretokenization_example import find_chunk_boundaries
from tqdm import tqdm

import multiprocessing as mp
import os
import re
import regex


PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Merge:
    """Represents a pair of tokens to be merged and the new merged token id."""
    def __init__(self, tokens_to_merge, new_token):
        self.tokens_to_merge = tokens_to_merge
        self.new_token = new_token


def get_num_workers():
    """Returns the number of CPU cores on the device."""
    return os.cpu_count()


def get_optimal_num_chunks():
    """Returns the optimal number of chunks for multiprocessing."""
    return get_num_workers() * 4


def pretokenize_chunk(chunk_boundaries, path, special_tokens):
    """Returns the map of pretokens to frequency for the file chunk.""" 
    start, end = chunk_boundaries
    count_by_pretoken = Counter()
    with open(path, 'r') as f:
        f.seek(start)
        data = f.read(end - start)
        data_parts = re.split("|".join([re.escape(token) for token in
                                        special_tokens]), data)
        for part in data_parts:
            for match in regex.finditer(PRETOKENIZATION_PATTERN, part):
                count_by_pretoken[match.group().encode('utf-8')] += 1
    return count_by_pretoken


def parallel_pretokenize(path, chunk_boundaries, special_tokens,):
    """Generates map of pretokens to frequency for entire file."""
    with mp.Pool(get_num_workers()) as pool:
        pretokenize = partial(pretokenize_chunk, path=path,
                              special_tokens=special_tokens)
        boundaries = ([(chunk_boundaries[i], chunk_boundaries[i + 1]) for i in
                      range(len(chunk_boundaries) - 1)])
        count_by_pretoken = pool.map(pretokenize, boundaries)
    return sum(count_by_pretoken, start=Counter())


 


def get_chunk_boundaries(num_items):
    optimal_num_chunks = get_optimal_num_chunks()
    if num_items < 10_000 * optimal_num_chunks:
        optimal_num_chunks = max(1, num_items // 10000)
    chunk_boundaries = []
    chunk_len = num_items // optimal_num_chunks
    for i in range(optimal_num_chunks):
        chunk_boundaries.append([i * chunk_len, (i + 1) * chunk_len])
    chunk_boundaries[-1][-1] = num_items
    return chunk_boundaries


def get_pair_stats(word_freqs):
    pair_freqs = Counter()
    for symbols, freq in word_freqs.items():
        for i in range(len(symbols) - 1):
            pair_freqs[symbols[i], symbols[i+1]] += freq
    return pair_freqs


def merge_symbols(best_pair, word_freqs):
    new_word_freqs = {}
    p1, p2 = best_pair
    new_symbol = p1 + p2
    for symbols, freq in word_freqs.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == p1 and symbols[i+1] == p2:
                new_symbols.append(new_symbol)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_word_freqs[tuple(new_symbols)] = freq
    return new_word_freqs


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if vocab_size < 256:
        raise ValueError("Vocabulary size must be at least 256.")

    vocab = {i: bytes([i]) for i in range(256)}
    for token_str in special_tokens:
        if token_str.encode("utf-8") not in vocab.values():
            vocab[len(vocab)] = token_str.encode("utf-8")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_chunks = [text]
    if special_tokens:
        special_pattern = f"({'|'.join(map(regex.escape, special_tokens))})"
        text_chunks = regex.split(special_pattern, text)
        text_chunks = [chunk for chunk in text_chunks if chunk]

    num_processes = os.cpu_count() or 1
    with mp.Pool(processes=num_processes) as pool:
        non_special_chunks = [chunk for chunk in text_chunks if chunk not in special_tokens]
        if non_special_chunks:
            chunk_freqs_list = pool.map(_process_chunk, non_special_chunks)
            total_word_freqs_str = Counter()
            for freqs in chunk_freqs_list:
                total_word_freqs_str.update(freqs)
        else:
            total_word_freqs_str = Counter()

    word_freqs_bytes = {word.encode("utf-8"): freq for word, freq in total_word_freqs_str.items()}
    word_freqs_symbols = {tuple(bytes([b]) for b in word): freq for word, freq in word_freqs_bytes.items()}

    merges = []
    num_merges_needed = vocab_size - len(vocab)
    
    for i in range(num_merges_needed):
        pair_stats = get_pair_stats(word_freqs_symbols)
        if not pair_stats:
            break

        best_pair = max(pair_stats, key=lambda p: (pair_stats[p], p))
        
        merges.append(best_pair)
        new_token_id = len(vocab)
        vocab[new_token_id] = best_pair[0] + best_pair[1]

        word_freqs_symbols = merge_symbols(best_pair, word_freqs_symbols)

    return vocab, merges


def _process_chunk(chunk_text: str) -> Counter:
    compiled_pat = regex.compile(PRETOKENIZATION_PATTERN)
    freqs = Counter()
    for match in compiled_pat.finditer(chunk_text):
        freqs[match.group(0)] += 1
    return freqs


if __name__ == "__main__":
    with profile_block("foo"):
        vocab, merges = train_bpe(
                input_path="/home/rylnaldo/Code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
                vocab_size=52000,
                special_tokens=["<|endoftext|>"])
        with open('./out/vocab.txt', 'w') as f:
            for token in vocab:
                f.write(f"{token}: {vocab[token].decode('utf-8')}\n")
        with open('./out/merges.txt', 'w') as f:
            for m1, m2 in merges:
                f.write(f"Merge {m1}, {m2}\n")

