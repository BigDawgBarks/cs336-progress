import os


def get_num_workers():
    """Returns the number of CPU cores on the device."""
    return os.cpu_count() or 1


def get_optimal_num_chunks():
    """Returns the optimal number of chunks for multiprocessing."""
    return get_num_workers() * 4


