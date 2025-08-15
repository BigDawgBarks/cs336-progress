from cs336_basics import util
from tqdm import tqdm

import multiprocessing as mp
import os
import re
import regex


PRETOKENIZATION_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
MAX_BYTES_PER_READ = 10_000_000 # each byte seems to take 10x overhead; aim for 2GiB per read at most.
MIN_CHUNK_SIZE = 10_000


def _process_chunk(chunk_text: str) -> Counter:
    compiled_pat = regex.compile(PRETOKENIZATION_PATTERN)
    freqs = Counter()
    for match in compiled_pat.finditer(chunk_text):
        freqs[match.group(0)] += 1
    return freqs


def pretokenize(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes], int]
    """Returns a mapping of tuple(bytes) to frequency for each pretoken."""


