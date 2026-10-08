"""Minimal reader for the training corpus: every series, shuffled uniformly, yielded in batches.

    unzip corpus.zip -d corpus     # -> corpus/corpus.npz
    for batch in batches("corpus", batch_size=64):
        ...  # list of 1-D float32 arrays of different lengths

corpus.npz holds two arrays: ``values`` (float32, all series concatenated) and ``offsets`` (int64; series i is
values[offsets[i]:offsets[i + 1]]). It is stored uncompressed, so ``values`` is memory-mapped, not read into RAM
(a plain ``np.load(path)["values"]`` would read all 8.9 GB).
"""

import zipfile
from pathlib import Path

import numpy as np


def _mmap(npz, name):
    """Memory-map one array of an uncompressed .npz."""
    with zipfile.ZipFile(npz) as archive:
        info = archive.getinfo(name)
        with archive.open(info) as member:
            fmt = np.lib.format
            read = fmt.read_array_header_1_0 if fmt.read_magic(member) == (1, 0) else fmt.read_array_header_2_0
            shape, _, dtype = read(member)
            header = member.tell()
    with open(npz, "rb") as handle:  # the member's data starts after its local file header
        handle.seek(info.header_offset + 26)
        name_len, extra_len = np.frombuffer(handle.read(4), np.uint16)
    start = info.header_offset + 30 + int(name_len) + int(extra_len) + header
    return np.memmap(npz, dtype=dtype, mode="r", offset=start, shape=shape)


def load(root):
    """Return a function mapping an index to one series, and the number of series."""
    root = Path(root)
    npz = root / "corpus.npz" if root.is_dir() else root
    values = _mmap(npz, "values.npy")
    offsets = np.load(npz)["offsets"]

    def get(i):
        return np.asarray(values[offsets[i] : offsets[i + 1]])

    return get, len(offsets) - 1


def batches(root, batch_size=64, seed=0):
    """One epoch: every series exactly once, in a uniformly random order."""
    get, n = load(root)
    order = np.random.default_rng(seed).permutation(n)
    for start in range(0, n, batch_size):
        yield [get(i) for i in order[start : start + batch_size]]
