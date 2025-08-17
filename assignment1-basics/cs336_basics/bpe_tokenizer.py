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
import heapq


PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
MAX_BYTES_PER_READ = 50_000_000 # each byte seems to take 10x overhead; aim for 2GiB per read at most.
MIN_CHUNK_SIZE = 10_000


def get_num_workers():
    """Returns the number of CPU cores on the device."""
    return os.cpu_count() or 1


def get_optimal_num_chunks():
    """Returns the optimal number of chunks for multiprocessing."""
    return get_num_workers() * 4


def get_chunk_boundaries(num_items):
    optimal_num_chunks = get_optimal_num_chunks()
    if num_items < MIN_CHUNK_SIZE * optimal_num_chunks:
        optimal_num_chunks = max(1, num_items // MIN_CHUNK_SIZE)
    chunk_boundaries = []
    chunk_len = num_items // optimal_num_chunks
    for i in range(optimal_num_chunks):
        chunk_boundaries.append([i * chunk_len, (i + 1) * chunk_len])
    chunk_boundaries[-1][-1] = num_items
    return chunk_boundaries


def _process_word_chunk(word_chunk):
    pair_freqs = Counter()
    for symbols, freq in word_chunk:
        for i in range(len(symbols) - 1):
            pair_freqs[symbols[i], symbols[i+1]] += freq
    return pair_freqs


def get_pair_stats(word_freqs, pool=None):
    word_items = list(word_freqs.items())
    num_items = len(word_items)
    
    if num_items < MIN_CHUNK_SIZE or not pool:
        return _process_word_chunk(word_items)
    
    chunk_boundaries = get_chunk_boundaries(num_items)
    word_chunks = []
    for start, end in chunk_boundaries:
        word_chunks.append(word_items[start:end])
    
    chunk_results = pool.map(_process_word_chunk, word_chunks)
    
    final_pair_freqs = Counter()
    for c in chunk_results:
        final_pair_freqs.update(c)
    return final_pair_freqs


def build_pair_to_symbols_mapping(word_freqs_symbols):
    pair_to_symbols = defaultdict(set)
    for symbols in word_freqs_symbols:
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pair_to_symbols[pair].add(symbols)
    return pair_to_symbols


def merge_symbols(best_pair, word_freqs, pair_to_symbols_mapping, pair_stats):
    p1, p2 = best_pair
    new_symbol = p1 + p2
    
    candidate_symbols = pair_to_symbols_mapping.get(best_pair, set())
    
    to_remove = []
    to_add = []
    
    for symbols in candidate_symbols:
        if symbols not in word_freqs:
            continue
        
        freq = word_freqs[symbols]
        
        if p1 not in symbols or p2 not in symbols:
            continue
        
        for i in range(len(symbols) - 1):
            old_pair = (symbols[i], symbols[i+1])
            pair_stats[old_pair] -= freq
            if pair_stats[old_pair] <= 0:
                del pair_stats[old_pair]
        
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == p1 and symbols[i+1] == p2:
                new_symbols.append(new_symbol)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        
        new_symbols_tuple = tuple(new_symbols)
        to_remove.append(symbols)
        to_add.append((new_symbols_tuple, freq))
        
        for i in range(len(new_symbols) - 1):
            new_pair = (new_symbols[i], new_symbols[i+1])
            pair_stats[new_pair] = pair_stats.get(new_pair, 0) + freq
        
        for i in range(len(new_symbols) - 1):
            pair = (new_symbols[i], new_symbols[i+1])
            pair_to_symbols_mapping[pair].add(new_symbols_tuple)
    
    for symbols in to_remove:
        del word_freqs[symbols]
    for symbols, freq in to_add:
        word_freqs[symbols] = freq


def get_file_size(input_path):
    with open(input_path, 'rb') as f:
        f.seek(0, 2)
        total_file_size = f.tell()
    return total_file_size


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if vocab_size < 256:
        raise ValueError("Vocabulary size must be at least 256.")

    vocab = {i: bytes([i]) for i in range(256)}
    for token_str in special_tokens:
        if token_str.encode("utf-8") not in vocab.values():
            vocab[len(vocab)] = token_str.encode("utf-8")

    bytes_processed = 0
    total_word_freqs_str = Counter()
    total_file_size = get_file_size(input_path)
    with tqdm(total=total_file_size, desc="Read", unit="B", unit_scale=True) as pbar:
        with mp.Pool(processes=get_num_workers()) as pool:
            while True:
                with open(input_path, "r", encoding="utf-8") as f:
                    f.seek(bytes_processed)
                    text = f.read(MAX_BYTES_PER_READ)
        
                if not text:
                    break
            
                chunk_start_pos = bytes_processed
                
                text_chunks = [text]
                if special_tokens:
                    special_pattern = f"({'|'.join(map(regex.escape, special_tokens))})"
                    text_chunks = regex.split(special_pattern, text)
                    text_chunks = [chunk for chunk in text_chunks if chunk]
        
                text_bytes = len(text.encode("utf-8"))
                if text_bytes < MAX_BYTES_PER_READ or len(text_chunks) <= 1:
                    bytes_processed += text_bytes
                    pbar.update(text_bytes)
                else:
                    last_chunk = text_chunks.pop()
                    bytes_processed += text_bytes - len(last_chunk.encode("utf-8"))
                    pbar.update(text_bytes - len(last_chunk.encode("utf-8")))
            
                non_special_chunks = [chunk for chunk in text_chunks if chunk not in special_tokens]
                if non_special_chunks:
                    chunk_args = []
                    current_offset = 0
                    for chunk in text_chunks:
                        chunk_bytes = len(chunk.encode("utf-8"))
                        if chunk not in special_tokens:
                            chunk_args.append((input_path, chunk_start_pos + current_offset, chunk_bytes))
                        current_offset += chunk_bytes
                    
                    chunk_freqs_list = pool.map(_process_file_chunk, chunk_args)
                    for freqs in chunk_freqs_list:
                        total_word_freqs_str.update(freqs)
            print("Building initial pair statistics...")
            word_freqs_bytes = {word.encode("utf-8"): freq for word, freq in total_word_freqs_str.items()}
            word_freqs_symbols = {tuple(bytes([b]) for b in word): freq for word, freq in word_freqs_bytes.items()}
            pair_stats = get_pair_stats(word_freqs_symbols, pool=pool)
            pair_to_symbols_mapping = build_pair_to_symbols_mapping(word_freqs_symbols)        
            print("Done building initial pair statistics.")


    merges = []
    num_merges_needed = vocab_size - len(vocab)
    
    for i in tqdm(range(num_merges_needed), desc="Merges", total=num_merges_needed):
        if not pair_stats:
            break

        best_pair = max(pair_stats, key=lambda p: (pair_stats[p], p))
        
        merges.append(best_pair)
        new_token_id = len(vocab)
        vocab[new_token_id] = best_pair[0] + best_pair[1]

        merge_symbols(best_pair, word_freqs_symbols, pair_to_symbols_mapping, pair_stats)

    return vocab, merges


def _process_file_chunk(args) -> Counter:
    file_path, start_byte, num_bytes = args
    compiled_pat = regex.compile(PRETOKENIZATION_PATTERN)
    freqs = Counter()
    with open(file_path, 'rb') as f:
        f.seek(start_byte)
        chunk_bytes = f.read(num_bytes)
        chunk_text = chunk_bytes.decode('utf-8')
    for match in compiled_pat.finditer(chunk_text):
        freqs[match.group(0)] += 1
    return freqs


if __name__ == "__main__":
    with profile_block("foo"):
        vocab, merges = train_bpe(
                # input_path="/home/rylnaldo/Code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
                # input_path="/home/rylnaldo/Code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
                input_path="/home/rylnaldo/Code/cs336/assignment1-basics/data/owt_train.txt",
                vocab_size=32_000,
                special_tokens=["<|endoftext|>"])
        with open('./out/vocab.txt', 'w') as f:
            for token in vocab:
                try:
                    decoded = vocab[token].decode('utf-8')
                except UnicodeDecodeError:
                    decoded = repr(vocab[token])
                f.write(f"{token}: {decoded}\n")
        with open('./out/merges.txt', 'w') as f:
            for m1, m2 in merges:
                f.write(f"Merge {m1}, {m2}\n")

