"""Train/val/test split assignment and the train negative cap. Pure functions, no I/O.

Blocks strategy: within a country, images are grouped by
``numeric_id // block_size`` and whole groups are assigned to a split, so
neighbouring ids (possibly neighbouring frames) never straddle splits except at
block boundaries. Groups are ordered by ``sha256(f"{seed}:{country}:{block}")``;
the first ``n_val`` go to val, the next ``n_test`` to test, the rest to train.

Similarity strategy (for datasets whose ids are not in capture order): within a
country, items whose 64-bit perceptual hashes are within ``max_bits`` Hamming
distance are linked (single linkage, so chains merge); each connected component
is one group. Component sizes are very unequal (a few giant components plus many
singletons), so quotas are filled by IMAGE count, not group count: components are
visited in ``sha256`` order and placed in val, then test, if they still fit the
remaining quota, otherwise in train. Over-merging only makes groups bigger, so it
is harmless for validity; under-merging is the leak.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Iterable, Mapping, Sequence

import numpy as np

SPLITS = ("train", "val", "test")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _half_up(x: float) -> int:
    return math.floor(x + 0.5)


def block_index(numeric_id: int, block_size: int) -> int:
    return numeric_id // block_size


def block_group_id(country: str, block: int) -> str:
    return f"{country}:b{block}"


def group_quotas(n_groups: int, f_val: float, f_test: float) -> tuple[int, int, int, bool]:
    """Return (n_train, n_val, n_test, reduced).

    val and test each get ``max(1, round_half_up(n * fraction))`` groups when
    their fraction is > 0. If that would leave no group for train, test is
    reduced first, then val (priority train > val > test); ``reduced`` is True
    when the result differs from the request.
    """
    want_val = max(1, _half_up(n_groups * f_val)) if f_val > 0 else 0
    want_test = max(1, _half_up(n_groups * f_test)) if f_test > 0 else 0
    n_val, n_test = want_val, want_test
    while n_groups - n_val - n_test < 1 and (n_val or n_test):
        if n_test:
            n_test -= 1
        else:
            n_val -= 1
    if n_groups == 0:
        return 0, 0, 0, False
    return n_groups - n_val - n_test, n_val, n_test, (n_val, n_test) != (want_val, want_test)


def assign_block_splits(
    country: str, blocks: Iterable[int], seed: int, f_val: float, f_test: float
) -> tuple[dict[int, str], list[str]]:
    """Assign each distinct block of one country to a split. Returns (mapping, warnings)."""
    unique = sorted(set(blocks))
    ordered = sorted(unique, key=lambda b: _digest(f"{seed}:{country}:{b}"))
    n_train, n_val, n_test, reduced = group_quotas(len(unique), f_val, f_test)
    mapping = {}
    for rank, block in enumerate(ordered):
        mapping[block] = "val" if rank < n_val else "test" if rank < n_val + n_test else "train"
    warnings = []
    if reduced:
        warnings.append(
            f"{country}: only {len(unique)} block(s); val/test were reduced to "
            f"{n_val}/{n_test} group(s) so train keeps at least one"
        )
    return mapping, warnings


# --------------------------------------------------------------------------- similarity

# Upper bound on the elements of one (rows x columns) work array: 4 M uint64 = 32 MB,
# so peak memory with a few temporaries stays far below 500 MB for any n.
_CHUNK_ELEMENTS = 1 << 22

_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)
_BYTE_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount64(x: np.ndarray) -> np.ndarray:
    """Per-element population count of a uint64 array (SWAR); the input is not modified."""
    x = x - ((x >> np.uint64(1)) & _M1)
    x = (x & _M2) + ((x >> np.uint64(2)) & _M2)
    x = (x + (x >> np.uint64(4))) & _M4
    return (x * _H01) >> np.uint64(56)


def _chunk_rows(n: int, per_row_factor: int = 1) -> int:
    return max(1, _CHUNK_ELEMENTS // max(1, n * per_row_factor))


def _hash_array(values: Sequence[int]) -> np.ndarray:
    return np.array([int(v) for v in values], dtype=np.uint64)


def similarity_components(hashes: Sequence[int | None], max_bits: int) -> list[int]:
    """Single-linkage components of the items of ONE country.

    Two items are linked when the Hamming distance of their 64-bit hashes is
    ``<= max_bits``; all pairs are compared exactly (vectorised, chunked). An item
    whose hash is None is its own component. Returns one label per input item; the
    label is the smallest input index in the component, so equal labels mean the
    same component. The partition does not depend on the order of ``hashes``.
    """
    if max_bits < 0:
        raise ValueError(f"max_bits must be >= 0, got {max_bits}")
    n = len(hashes)
    present = [i for i, h in enumerate(hashes) if h is not None]
    parent = list(range(n))

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    m = len(present)
    if m > 1:
        arr = _hash_array([hashes[i] for i in present])  # type: ignore[misc]
        index = np.array(present, dtype=np.int64)
        step = _chunk_rows(m)
        limit = np.uint64(max_bits)
        for start in range(0, m, step):
            stop = min(m, start + step)
            dist = _popcount64(arr[start:stop, None] ^ arr[None, start:])
            rows, cols = np.nonzero(dist <= limit)
            cols = cols + start
            keep = cols > rows + start  # upper triangle only
            for a, b in zip(index[rows[keep] + start].tolist(), index[cols[keep]].tolist()):
                ra, rb = find(a), find(b)
                if ra != rb:  # the smaller index always becomes the root
                    if ra < rb:
                        parent[rb] = ra
                    else:
                        parent[ra] = rb
    return [find(i) for i in range(n)]


def find_cross_split_near_pair(
    hashes: Sequence[int | None], split_labels: Sequence[str], max_bits: int
) -> tuple[int, int, int] | None:
    """First pair (i, j, distance), i < j, in different splits with distance <= max_bits.

    Deliberately independent of ``similarity_components`` (different popcount, no
    union-find): it works only from the hashes and the final split labels. Items with
    a None hash are ignored. Returns None when no such pair exists.
    """
    if len(hashes) != len(split_labels):
        raise ValueError("hashes and split_labels must have the same length")
    present = [i for i, h in enumerate(hashes) if h is not None]
    m = len(present)
    if m < 2:
        return None
    arr = _hash_array([hashes[i] for i in present])  # type: ignore[misc]
    codes = {s: k for k, s in enumerate(sorted(set(split_labels)))}
    lab = np.array([codes[split_labels[i]] for i in present], dtype=np.int64)
    step = _chunk_rows(m, per_row_factor=8)
    for start in range(0, m, step):
        stop = min(m, start + step)
        xor = np.ascontiguousarray(arr[start:stop, None] ^ arr[None, start:])
        bits = _BYTE_POPCOUNT[xor.view(np.uint8)].reshape(*xor.shape, 8)
        dist = bits.sum(axis=2, dtype=np.int64)
        different = lab[start:stop, None] != lab[None, start:]
        rows, cols = np.nonzero((dist <= max_bits) & different)
        if rows.size:
            cols = cols + start
            keep = cols > rows + start
            if keep.any():
                r = int(rows[keep][0]) + start
                c = int(cols[keep][0])
                return present[r], present[c], int(dist[r - start, c - start])
    return None


def similarity_group_id(country: str, min_numeric_id: int) -> str:
    return f"{country}:c{min_numeric_id:06d}"


def component_targets(n_images: int, f_val: float, f_test: float) -> tuple[int, int]:
    """(val_target, test_target) image counts for a country with ``n_images`` items."""
    return _half_up(n_images * f_val), _half_up(n_images * f_test)


def assign_component_splits(
    country: str,
    components: Mapping[Hashable, int],
    seed: int,
    f_val: float,
    f_test: float,
) -> tuple[dict[Hashable, str], list[str]]:
    """Assign components ({key: n_images}) of one country to splits by image-count quota.

    Components are visited in ascending ``sha256(f"{seed}:{country}:{key}")`` order
    (ties broken by key). A component goes to val if it still fits the val target,
    else to test if it fits the test target, else to train, so a component larger
    than a remaining quota stays in train while later, smaller ones can still fill
    val/test. Warns when val or test ends below 80% of its (non-zero) target.
    Returns (mapping, warnings).
    """
    n_images = sum(components.values())
    val_target, test_target = component_targets(n_images, f_val, f_test)
    ordered = sorted(components, key=lambda k: (_digest(f"{seed}:{country}:{k}"), k))
    val_count = test_count = 0
    mapping: dict[Hashable, str] = {}
    for key in ordered:
        size = components[key]
        if val_count + size <= val_target:
            mapping[key] = "val"
            val_count += size
        elif test_count + size <= test_target:
            mapping[key] = "test"
            test_count += size
        else:
            mapping[key] = "train"
    warnings = []
    for name, got, target in (("val", val_count, val_target), ("test", test_count, test_target)):
        if target > 0 and got < 0.8 * target:
            state = "empty" if got == 0 else f"{got}/{target} images"
            warnings.append(
                f"{country}: {name} split is {state} (below 80% of its target of {target}); "
                "large near-duplicate components could not be placed"
            )
    return mapping, warnings


def assign_country_splits(
    countries: Sequence[str], val_countries: Sequence[str], test_countries: Sequence[str]
) -> dict[str, str]:
    return {
        c: "val" if c in val_countries else "test" if c in test_countries else "train"
        for c in countries
    }


def negative_cap(n_train_positives: int, ratio: float) -> int:
    """Maximum number of train negatives: floor(ratio * positives)."""
    return math.floor(ratio * n_train_positives + 1e-9)


def select_capped_negatives(source_ids: Iterable[str], n_keep: int, seed: int) -> set[str]:
    """Keep ``n_keep`` negatives, chosen by ascending ``sha256(f"{seed}:{source_id}")``."""
    ordered = sorted(set(source_ids), key=lambda s: (_digest(f"{seed}:{s}"), s))
    return set(ordered[:n_keep])
