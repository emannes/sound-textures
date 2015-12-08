import os
from common import sound_path, compressed_path

def compression_rate(base_name):
    uncompressed_size = os.path.getsize(sound_path(base_name))
    compressed_size = os.path.getsize(compressed_path(base_name))
    assert compressed_size > 0
    return float(uncompressed_size)/compressed_size
