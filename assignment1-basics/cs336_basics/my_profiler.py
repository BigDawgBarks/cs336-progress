import time
import psutil
import os
import cProfile
import pstats
import io
from contextlib import contextmanager

@contextmanager
def profile_block(name="Code block"):
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    start_mem = process.memory_info().rss / 1024 / 1024
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        yield
    finally:
        profiler.disable()
        end_time = time.perf_counter()
        end_mem = process.memory_info().rss / 1024 / 1024
        
        print(f"\n--- Performance Stats for {name} ---")
        print(f"Runtime: {end_time - start_time:.3f} seconds")
        print(f"Memory change: {end_mem - start_mem:.2f} MB")
        print(f"Peak memory: {end_mem:.2f} MB")
        
        print(f"\n--- CPU Profile (Top 10) ---")
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        print(s.getvalue())

