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


def count_chunk_pairs(chunk, pair_to_merge: Merge):
    """
    Count token pair frequencies for a single chunk.

    If the pair_to_merge is None, count all token pairs. If populated with a
    pair (tok1, tok2), return only the count of pairs adjacent to the newly
    minted token after the merge.

    Args:
        pair_to_merge: either None or a pair of tokens to merge.
        chunk: list of tuples containing (pretoken bytestring, tokens in
            pretoken, number of times this pretoken appears)

    Returns:
        count_by_token_pair: map of (tok1, tok2) -> count
        pretokens_by_token_pair: map of (tok1, tok2) -> pretokens containing
            the pair
        tokens_by_pretoken: map of pretoken bytestring -> new sequence of
            tokens
    """
    count_by_token_pair = Counter()
    pretokens_by_token_pair = defaultdict(set)
    tokens_by_pretoken = dict()
    for pretoken, tokens_sequence, pretoken_count in chunk:
        if pair_to_merge:
            found = False
            first, second = pair_to_merge.tokens_to_merge
            new_token = pair_to_merge.new_token
            new_tokens_sequence = []
            i = 0
            while i < len(tokens_sequence):
                if (i < len(tokens_sequence) - 1 and tokens_sequence[i] ==
                    first and tokens_sequence[i + 1] == second):
                    if not found:
                        found = True
                        new_tokens_sequence.extend(tokens_sequence[:i])
                    new_tokens_sequence.append(new_token)
                    if i > 0:
                        left_pair = (tokens_sequence[i - 1], new_token)
                        count_by_token_pair[left_pair] += pretoken_count
                        pretokens_by_token_pair[left_pair].add(pretoken)
                    if i < len(tokens_sequence) - 2:
                        right_pair = (new_token, tokens_sequence[i + 1])
                        count_by_token_pair[right_pair] += pretoken_count
                        pretokens_by_token_pair[right_pair].add(pretoken) 
                    i += 2
                else:
                    if found:
                        new_tokens_sequence.append(tokens_sequence[i])
                    i += 1
            if found:
                tokens_by_pretoken[pretoken] = new_tokens_sequence
        else: # no pair merge 
            for i in range(len(tokens_sequence) - 1):
                first, second = tokens_sequence[i], tokens_sequence[i + 1]
                count_by_token_pair[(first, second)] += pretoken_count
                pretokens_by_token_pair[(first, second)].add(pretoken)

    return count_by_token_pair, pretokens_by_token_pair, tokens_by_pretoken 


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


def parallel_train_bpe_tokenizer(pretokens, vocab_size, special_tokens):
    """
    Given a set of pre-tokens, train a BPE tokenizer.

    Args:
        pretokens: dictionary mapping byte sequences (pre-tokens) to frequencies
        vocab_size: positive integer that defines max final vocab size
        special_tokens: list of strings to add to the vocab. Does not otherwise
            affect BPE training.

    Returns:
        vocab: tokenizer vocabulary mapping from int (token ID) to bytes
        merges: list of BPE merges produced by training, tuple (<token1>,
            <token2>)
    """
    # vocab maps int -> token byte sequence
    vocab = dict([(i, chr(i).encode('utf-8')) for i in range(256)])
    merges = []
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    
    tokens_by_pretoken = dict()
    for pretoken in pretokens:
        tokens_by_pretoken[pretoken] = ([b for b in pretoken])

    pair_to_merge = None
    total_count_by_token_pair = Counter()
    pretokens_by_token_pair = defaultdict(set)
    pretokens_list = list(pretokens.items())

    # Progress tracking setup
    initial_vocab_size = len(vocab)
    total_merges = vocab_size - initial_vocab_size
    print(f"Starting BPE training: {initial_vocab_size} → {vocab_size} tokens \
          ({total_merges} merges)")

    with tqdm(total=total_merges, desc="BPE Training") as pbar:
        with mp.Pool(get_num_workers()) as pool:
            while len(vocab) < vocab_size:
                pretoken_chunk_boundaries = get_chunk_boundaries(len(pretokens_list))
                chunks = []
                for start, end in pretoken_chunk_boundaries:
                    chunks.append([])
                    for i in range(start, end):
                        pretoken, count = pretokens_list[i]
                        tokens = tokens_by_pretoken[pretoken]
                        chunks[-1].append((pretoken, tokens, count))
                count_pairs = partial(count_chunk_pairs,
                                      pair_to_merge=pair_to_merge)
                chunk_results = pool.map(count_pairs, chunks)
        
                (count_by_token_pair_list, pretokens_by_token_pair_list,
                    tokens_by_pretoken_list) = zip(*chunk_results)
                # Update token pair counts
                total_count_by_token_pair = sum(count_by_token_pair_list,
                                                start=total_count_by_token_pair)
                if pair_to_merge:
                    del total_count_by_token_pair[pair_to_merge.tokens_to_merge]
                # Update mapping of token pairs -> pretokens
                for chunk_pretokens_by_token_pair in pretokens_by_token_pair_list:
                    for pair in chunk_pretokens_by_token_pair:
                        pretokens_by_token_pair[pair].update(chunk_pretokens_by_token_pair[pair])
                if pair_to_merge:
                    del pretokens_by_token_pair[pair_to_merge.tokens_to_merge]
                # Update pretoken -> token list mappings
                for chunk_tokens_by_pretoken in tokens_by_pretoken_list:
                    for pretoken in chunk_tokens_by_pretoken:
                        tokens_by_pretoken[pretoken] = chunk_tokens_by_pretoken[pretoken]
                
                max_count = 0
                most_common_pair = None
                for token_pair in total_count_by_token_pair:
                    if total_count_by_token_pair[token_pair] > max_count:
                        max_count = total_count_by_token_pair[token_pair]
                        most_common_pair = token_pair
                    elif (total_count_by_token_pair[token_pair] == max_count and
                          most_common_pair < token_pair):
                        most_common_pair = token_pair # break ties lexicographically
        
                v1, v2 = most_common_pair
                vocab[len(vocab)] = vocab[v1] + vocab[v2]
                pair_to_merge = Merge(most_common_pair, len(vocab) - 1)
                merges.append((vocab[v1], vocab[v2]))
                pretokens_list = ([(pretoken, pretokens[pretoken]) for pretoken in
                                  pretokens_by_token_pair[most_common_pair]])
                    
                pbar.update(1)
                pbar.set_postfix(vocab_size=len(vocab), max_count=max_count)

    return vocab, merges


def train_bpe(input_path: str, vocab_size: int, special_tokens:
                        list[str]):
    """
    Trains BPE tokenizer.

    Args:
        input_path: path to text file with training data
        vocab_size: positive integer that defines max final vocab vocab_size
        special_tokens: list of strings to add to the vocab. Does not otherwise
            affect BPE training.

    Returns:
        vocab: tokenizer vocabulary mapping from int (token ID) to bytes
        merges: list of BPE merges produced by training, tuple (<token1>,
            <token2>)
    """
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab size is not large enough; must be at least 256 \
                         + number of special tokens")

    optimal_num_chunks = get_optimal_num_chunks()
    with open(input_path, 'rb') as f:
        chunk_boundaries = find_chunk_boundaries(f, optimal_num_chunks,
                                                 "<|endoftext|>".encode("utf-8"))
    pretokens = parallel_pretokenize(input_path, chunk_boundaries, special_tokens)

    return parallel_train_bpe_tokenizer(pretokens, vocab_size, special_tokens)


if __name__ == "__main__":
    with profile_block("My computation"):
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

