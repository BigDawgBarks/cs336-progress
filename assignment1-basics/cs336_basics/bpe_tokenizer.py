"""Logic for Training BPE Tokenizer."""

from collections import defaultdict, Counter
from functools import partial
from cs336_basics.my_profiler import profile_block
from cs336_basics.pretokenization_example import find_chunk_boundaries
from tqdm import tqdm

import multiprocessing as mp
import os
import pickle
import re
import regex
import heapq
import tempfile


PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
MAX_BYTES_PER_READ = 50_000_000 # each byte seems to take 10x overhead; aim for 2GiB per read at most.
MIN_CHUNK_SIZE = 10_000
CHUNKS_PER_BATCH = 300


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


def save_checkpoint(checkpoint_path, merge_index, vocab, merges, word_freqs_symbols, pair_stats, pair_to_symbols_mapping):
    checkpoint_data = {
        'merge_index': merge_index,
        'vocab': vocab,
        'merges': merges,
        'word_freqs_symbols': word_freqs_symbols,
    }
    
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=os.path.dirname(checkpoint_path)) as tmp_file:
        pickle.dump(checkpoint_data, tmp_file)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_path = tmp_file.name
    
    os.rename(temp_path, checkpoint_path)


def load_checkpoint(checkpoint_path, pool=None):
    if not os.path.exists(checkpoint_path):
        return None
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    merge_index = checkpoint_data['merge_index']
    vocab = checkpoint_data['vocab']
    merges = checkpoint_data['merges']
    word_freqs_symbols = checkpoint_data['word_freqs_symbols']
    
    # Check if this is an old checkpoint with pair_stats and pair_to_symbols_mapping
    if 'pair_stats' in checkpoint_data and 'pair_to_symbols_mapping' in checkpoint_data:
        # Backwards compatibility: use saved values
        pair_stats = checkpoint_data['pair_stats']
        pair_to_symbols_mapping = checkpoint_data['pair_to_symbols_mapping']
    else:
        # New format: rebuild from word_freqs_symbols
        print("Rebuilding pair statistics from checkpoint data...")
        pair_stats = get_pair_stats(word_freqs_symbols, pool=pool)
        pair_to_symbols_mapping = build_pair_to_symbols_mapping(word_freqs_symbols)
        print("Done rebuilding pair statistics.")
    
    return (
        merge_index,
        vocab,
        merges,
        word_freqs_symbols,
        pair_stats,
        pair_to_symbols_mapping
    )


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], checkpoint_path: str = None, checkpoint_frequency: int = 2000) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if vocab_size < 256:
        raise ValueError("Vocabulary size must be at least 256.")

    with mp.Pool(processes=get_num_workers()) as pool:
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint_data = load_checkpoint(checkpoint_path, pool=pool)
            if checkpoint_data:
                start_merge_index, vocab, merges, word_freqs_symbols, pair_stats, pair_to_symbols_mapping = checkpoint_data
                print(f"Resuming from merge {start_merge_index}")
            else:
                start_merge_index = 0
                vocab = {i: bytes([i]) for i in range(256)}
                for token_str in special_tokens:
                    if token_str.encode("utf-8") not in vocab.values():
                        vocab[len(vocab)] = token_str.encode("utf-8")
        else:
            start_merge_index = 0
            vocab = {i: bytes([i]) for i in range(256)}
            for token_str in special_tokens:
                if token_str.encode("utf-8") not in vocab.values():
                    vocab[len(vocab)] = token_str.encode("utf-8")

        if start_merge_index == 0:
            bytes_processed = 0
            total_word_freqs_str = Counter()
            total_file_size = get_file_size(input_path)
            with tqdm(total=total_file_size, desc="Read", unit="B", unit_scale=True) as pbar:
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
                        chunk_infos = []
                        current_offset = 0
                        for chunk in text_chunks:
                            chunk_bytes = len(chunk.encode("utf-8"))
                            if chunk not in special_tokens:
                                chunk_infos.append((chunk_start_pos + current_offset, chunk_bytes))
                            current_offset += chunk_bytes
                        
                        batch_args = []
                        for i in range(0, len(chunk_infos), CHUNKS_PER_BATCH):
                            batch = chunk_infos[i:i + CHUNKS_PER_BATCH]
                            batch_args.append((input_path, batch))

                        chunk_freqs_list = pool.map(_process_file_chunk_batch, batch_args)
                        for freqs in chunk_freqs_list:
                            total_word_freqs_str.update(freqs)

            print("Building initial pair statistics...")
            word_freqs_bytes = {word.encode("utf-8"): freq for word, freq in total_word_freqs_str.items()}
            word_freqs_symbols = {tuple(bytes([b]) for b in word): freq for word, freq in word_freqs_bytes.items()}
            pair_stats = get_pair_stats(word_freqs_symbols, pool=pool)
            pair_to_symbols_mapping = build_pair_to_symbols_mapping(word_freqs_symbols)        
            print("Done building initial pair statistics.")
            merges = []

        initial_vocab_size = 256 + len(special_tokens)
        num_merges_needed = vocab_size - initial_vocab_size
        
        for i in tqdm(range(start_merge_index, num_merges_needed), desc="Merges", total=num_merges_needed, initial=start_merge_index):
            if not pair_stats:
                break

            best_pair = max(pair_stats, key=lambda p: (pair_stats[p], p))
            
            merges.append(best_pair)
            new_token_id = len(vocab)
            vocab[new_token_id] = best_pair[0] + best_pair[1]

            merge_symbols(best_pair, word_freqs_symbols, pair_to_symbols_mapping, pair_stats)
            
            if checkpoint_path and (i + 1) % checkpoint_frequency == 0:
                save_checkpoint(checkpoint_path, i + 1, vocab, merges, word_freqs_symbols, pair_stats, pair_to_symbols_mapping)
                print(f"Checkpoint saved at merge {i + 1}")

        if checkpoint_path:
            save_checkpoint(checkpoint_path, num_merges_needed, vocab, merges, word_freqs_symbols, pair_stats, pair_to_symbols_mapping)
            print("Final checkpoint saved")

        return vocab, merges


def _process_file_chunk_batch(args) -> Counter:
    file_path, chunk_infos = args
    compiled_pat = regex.compile(PRETOKENIZATION_PATTERN)
    freqs = Counter()

    with open(file_path, 'rb') as f:
        for start_byte, num_bytes in chunk_infos:
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
                special_tokens=["<|endoftext|>"],
                checkpoint_path="./out/owt_train_bpe_checkpoint.pkl",
                checkpoint_frequency=1000)
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

