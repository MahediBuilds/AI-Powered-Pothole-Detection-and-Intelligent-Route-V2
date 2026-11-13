import pytest

from roadsense.data import splits


def test_group_quotas_examples():
    assert splits.group_quotas(10, 0.1, 0.1) == (8, 1, 1, False)
    assert splits.group_quotas(25, 0.1, 0.1) == (19, 3, 3, False)  # 2.5 rounds half up
    assert splits.group_quotas(3, 0.1, 0.1) == (1, 1, 1, False)
    assert splits.group_quotas(2, 0.1, 0.1) == (1, 1, 0, True)  # test is sacrificed first
    assert splits.group_quotas(1, 0.1, 0.1) == (1, 0, 0, True)
    assert splits.group_quotas(5, 0.2, 0.0) == (4, 1, 0, False)  # zero fraction gets none
    assert splits.group_quotas(0, 0.1, 0.1) == (0, 0, 0, False)


def test_block_assignment_counts_and_determinism():
    blocks = list(range(20)) * 5  # repeated ids collapse to distinct blocks
    a, _ = splits.assign_block_splits("India", blocks, 42, 0.1, 0.1)
    b, _ = splits.assign_block_splits("India", reversed(blocks), 42, 0.1, 0.1)
    assert a == b
    assert sorted(a) == list(range(20))
    counts = {s: sum(1 for v in a.values() if v == s) for s in splits.SPLITS}
    assert counts == {"train": 16, "val": 2, "test": 2}


def test_different_seed_changes_assignment():
    a, _ = splits.assign_block_splits("India", range(40), 1, 0.1, 0.1)
    b, _ = splits.assign_block_splits("India", range(40), 2, 0.1, 0.1)
    assert a != b


def test_country_is_part_of_the_hash():
    a, _ = splits.assign_block_splits("India", range(40), 42, 0.1, 0.1)
    b, _ = splits.assign_block_splits("Japan", range(40), 42, 0.1, 0.1)
    assert a != b


def test_few_blocks_give_val_and_test_at_least_one_when_possible():
    mapping, warnings = splits.assign_block_splits("Czech", [0, 1, 2], 42, 0.1, 0.1)
    assert sorted(mapping.values()) == ["test", "train", "val"]
    assert warnings == []


def test_too_few_blocks_prioritise_train_then_val_with_warning():
    mapping, warnings = splits.assign_block_splits("Czech", [0, 1], 42, 0.1, 0.1)
    assert sorted(mapping.values()) == ["train", "val"]
    assert len(warnings) == 1 and "Czech" in warnings[0]
    mapping, warnings = splits.assign_block_splits("Czech", [0], 42, 0.1, 0.1)
    assert list(mapping.values()) == ["train"] and warnings


def test_block_index_and_group_id():
    assert splits.block_index(0, 200) == 0
    assert splits.block_index(199, 200) == 0
    assert splits.block_index(200, 200) == 1
    assert splits.block_group_id("India", 3) == "India:b3"


def test_country_strategy():
    mapping = splits.assign_country_splits(
        ["India", "Japan", "Czech", "Norway"], ["Czech"], ["Norway"]
    )
    assert mapping == {"India": "train", "Japan": "train", "Czech": "val", "Norway": "test"}


@pytest.mark.parametrize(
    ("pos", "ratio", "expected"),
    [(3, 2.0, 6), (10, 0.29, 2), (100, 0.29, 29), (10, 1.1, 11), (0, 2.0, 0), (5, 0, 0)],
)
def test_negative_cap(pos, ratio, expected):
    assert splits.negative_cap(pos, ratio) == expected


def test_select_capped_negatives_is_deterministic_and_order_independent():
    ids = [f"India_{i:06d}" for i in range(50)]
    a = splits.select_capped_negatives(ids, 10, 42)
    b = splits.select_capped_negatives(reversed(ids), 10, 42)
    assert a == b and len(a) == 10 and a <= set(ids)
    assert splits.select_capped_negatives(ids, 10, 43) != a
    assert splits.select_capped_negatives(ids, 100, 42) == set(ids)
    assert splits.select_capped_negatives(ids, 0, 42) == set()


# ------------------------------------------------------------------ similarity strategy

import random  # noqa: E402
import time  # noqa: E402

FULL = (1 << 64) - 1


def partition(labels):
    groups: dict[int, set[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, set()).add(i)
    return {frozenset(g) for g in groups.values()}


def naive_labels(hashes, max_bits):
    """O(n^2) pure-Python single linkage, the reference for the numpy implementation."""
    n = len(hashes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] is None or hashes[j] is None:
                continue
            if bin(hashes[i] ^ hashes[j]).count("1") <= max_bits:
                parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def flip(value, positions):
    for p in positions:
        value ^= 1 << p
    return value


def test_chain_is_one_component_by_single_linkage():
    a = 0
    b = flip(a, range(0, 4))  # 4 bits from a
    c = flip(b, range(4, 8))  # 4 bits from b, 8 bits from a
    assert bin(a ^ c).count("1") == 8
    labels = splits.similarity_components([a, b, c], 4)
    assert len(set(labels)) == 1
    labels = splits.similarity_components([a, c], 4)  # without the middle link: apart
    assert len(set(labels)) == 2
    assert len(set(splits.similarity_components([a, b, c], 3))) == 3


def test_far_apart_items_are_separate_and_extreme_values_work():
    labels = splits.similarity_components([0, FULL], 16)
    assert labels[0] != labels[1]
    assert splits.similarity_components([FULL, FULL], 0) == [0, 0]


def test_none_hashes_are_singletons():
    labels = splits.similarity_components([None, 7, None, 7, None], 64)
    assert labels[1] == labels[3]  # the two 7s merge
    assert len({labels[0], labels[2], labels[4], labels[1]}) == 4  # every None is alone
    assert splits.similarity_components([None, None], 8) == [0, 1]


def test_degenerate_inputs():
    assert splits.similarity_components([], 8) == []
    assert splits.similarity_components([5], 8) == [0]
    with pytest.raises(ValueError):
        splits.similarity_components([1, 2], -1)


def test_threshold_zero_groups_only_identical_hashes():
    hashes = [10, 11, 10, 11, 10, 99]
    labels = splits.similarity_components(hashes, 0)
    assert partition(labels) == {frozenset({0, 2, 4}), frozenset({1, 3}), frozenset({5})}


def test_labels_are_smallest_member_index():
    assert splits.similarity_components([3, 900, 3, 900, 3], 0) == [0, 1, 0, 1, 0]


def test_partition_is_invariant_under_permutation():
    rng = random.Random(5)
    hashes = [rng.getrandbits(14) for _ in range(120)]  # dense space: many chance links
    base = partition(splits.similarity_components(hashes, 3))
    assert 1 < len(base) < len(hashes)  # the test is not vacuous
    for _ in range(5):
        order = list(range(len(hashes)))
        rng.shuffle(order)
        shuffled = splits.similarity_components([hashes[i] for i in order], 3)
        mapped = {frozenset(order[k] for k in g) for g in partition(shuffled)}
        assert mapped == base


@pytest.mark.parametrize("max_bits", [0, 3, 6, 10])
def test_numpy_matches_naive_reference(max_bits):
    rng = random.Random(11)
    hashes: list[int | None] = [rng.getrandbits(64) for _ in range(300)]
    for _ in range(25):  # planted near pairs, some chained
        src = rng.randrange(300)
        dst = rng.randrange(300)
        hashes[dst] = flip(hashes[src], rng.sample(range(64), rng.randrange(0, 11)))
    hashes[5] = hashes[6] = None
    assert partition(splits.similarity_components(hashes, max_bits)) == partition(
        naive_labels(hashes, max_bits)
    )
    # a dense low-entropy variant, where chance links create nontrivial components
    dense = [rng.getrandbits(20) << 30 for _ in range(300)]
    assert partition(splits.similarity_components(dense, max_bits)) == partition(
        naive_labels(dense, max_bits)
    )
    if max_bits == 3:
        assert 1 < len(partition(naive_labels(dense, 3))) < 300


def test_planted_pairs_are_actually_linked():
    rng = random.Random(3)
    hashes = [rng.getrandbits(64) for _ in range(50)]
    hashes[40] = flip(hashes[10], [1, 7, 30])
    labels = splits.similarity_components(hashes, 4)
    assert labels[40] == labels[10]
    assert len(set(labels)) == 49


def test_similarity_components_scale_12000_hashes():
    rng = random.Random(0)
    hashes = [rng.getrandbits(64) for _ in range(12000)]
    for i in range(0, 1200, 3):  # some planted links so union-find is exercised
        hashes[i + 1] = flip(hashes[i], [i % 64])
    start = time.perf_counter()
    labels = splits.similarity_components(hashes, 8)
    elapsed = time.perf_counter() - start
    print(f"similarity_components on 12000 hashes: {elapsed:.2f} s")
    assert elapsed < 120
    assert len(set(labels)) == 12000 - 400
    assert all(labels[i] == labels[i + 1] for i in range(0, 1200, 3))


def test_cross_split_pair_finder_is_independent_and_exact():
    rng = random.Random(8)
    hashes = [rng.getrandbits(64) for _ in range(80)]
    hashes[70] = flip(hashes[3], [5, 6])
    same = ["train"] * 80
    assert splits.find_cross_split_near_pair(hashes, same, 4) is None  # same split: fine
    labels = list(same)
    labels[70] = "val"
    assert splits.find_cross_split_near_pair(hashes, labels, 4) == (3, 70, 2)
    assert splits.find_cross_split_near_pair(hashes, labels, 1) is None  # beyond threshold
    hashes[10] = None
    labels[10] = "test"
    assert splits.find_cross_split_near_pair(hashes, labels, 0) is None  # None never matches
    assert splits.find_cross_split_near_pair([FULL], ["train"], 8) is None
    with pytest.raises(ValueError):
        splits.find_cross_split_near_pair([1, 2], ["train"], 8)


def test_cross_split_finder_agrees_with_naive_on_random_data():
    rng = random.Random(21)
    hashes = [rng.getrandbits(12) << 40 for _ in range(150)]
    labels = [rng.choice(splits.SPLITS) for _ in hashes]
    for bits in (0, 1, 2):
        naive = [
            (i, j)
            for i in range(150)
            for j in range(i + 1, 150)
            if labels[i] != labels[j] and bin(hashes[i] ^ hashes[j]).count("1") <= bits
        ]
        found = splits.find_cross_split_near_pair(hashes, labels, bits)
        assert (found is None) == (not naive)
        if found:
            assert (found[0], found[1]) == naive[0]


def keyed(sizes):
    return {(i, f"India_{i:06d}"): s for i, s in enumerate(sizes)}


def count_by_split(components, mapping):
    out = {s: 0 for s in splits.SPLITS}
    for key, size in components.items():
        out[mapping[key]] += size
    return out


def test_component_quotas_are_by_image_count():
    comps = keyed([1] * 100)
    mapping, warnings = splits.assign_component_splits("India", comps, 42, 0.1, 0.1)
    assert count_by_split(comps, mapping) == {"train": 80, "val": 10, "test": 10}
    assert warnings == []
    # a mix of sizes: counted by images, never above the target
    comps = keyed([5, 3, 2, 2, 1, 1, 1, 1] * 12 + [1] * 20)
    n = sum(comps.values())
    mapping, _ = splits.assign_component_splits("India", comps, 42, 0.1, 0.1)
    got = count_by_split(comps, mapping)
    assert got["val"] <= splits.component_targets(n, 0.1, 0.1)[0]
    assert got["test"] <= splits.component_targets(n, 0.1, 0.1)[1]
    assert got["val"] > 0 and got["test"] > 0 and sum(got.values()) == n


def test_giant_component_goes_to_train_and_small_ones_still_fill_val_test():
    comps = {(0, "India_000000"): 60, **{(i, f"India_{i:06d}"): 1 for i in range(1, 41)}}
    for seed in range(10):
        mapping, warnings = splits.assign_component_splits("India", comps, seed, 0.1, 0.1)
        assert mapping[(0, "India_000000")] == "train"  # 60 > val target 10 (n = 100)
        got = count_by_split(comps, mapping)
        assert got["val"] == 10 and got["test"] == 10
        assert warnings == []


def test_totals_never_exceed_targets_for_random_sizes():
    rng = random.Random(4)
    for trial in range(30):
        comps = keyed([rng.choice([1, 1, 1, 2, 3, 8, 40]) for _ in range(rng.randrange(1, 80))])
        n = sum(comps.values())
        f_val, f_test = rng.choice([(0.1, 0.1), (0.2, 0.0), (0.25, 0.25), (0.0, 0.0)])
        mapping, _ = splits.assign_component_splits("Czech", comps, trial, f_val, f_test)
        got = count_by_split(comps, mapping)
        vt, tt = splits.component_targets(n, f_val, f_test)
        assert got["val"] <= vt and got["test"] <= tt
        assert sum(got.values()) == n and set(mapping) == set(comps)


def test_component_assignment_is_deterministic_and_seed_dependent():
    comps = keyed([1] * 60)
    a, _ = splits.assign_component_splits("India", comps, 42, 0.1, 0.1)
    b, _ = splits.assign_component_splits("India", dict(reversed(list(comps.items()))), 42, 0.1, 0.1)
    c, _ = splits.assign_component_splits("India", comps, 43, 0.1, 0.1)
    assert a == b and a != c
    d, _ = splits.assign_component_splits("Japan", comps, 42, 0.1, 0.1)
    assert a != d  # the country is part of the hash


def test_component_order_follows_documented_digest():
    comps = keyed([1] * 20)  # val target 2, test target 2
    order = sorted(comps, key=lambda k: (splits._digest(f"42:India:{k}"), k))
    mapping, _ = splits.assign_component_splits("India", comps, 42, 0.1, 0.1)
    assert [mapping[k] for k in order[:4]] == ["val", "val", "test", "test"]
    assert all(mapping[k] == "train" for k in order[4:])


def test_warning_when_targets_cannot_be_met():
    comps = {(0, "India_000000"): 100}  # val/test targets are 10 images each; nothing fits
    mapping, warnings = splits.assign_component_splits("India", comps, 42, 0.1, 0.1)
    assert list(mapping.values()) == ["train"]
    assert len(warnings) == 2
    assert all(w.startswith("India:") for w in warnings)
    assert "val split is empty" in warnings[0] and "test split is empty" in warnings[1]
    # below 80% but not empty: target 10, achieved 7
    comps = {(0, "a"): 7, (1, "b"): 93}
    mapping, warnings = splits.assign_component_splits("India", comps, 42, 0.1, 0.0)
    assert mapping[(0, "a")] == "val" and mapping[(1, "b")] == "train"
    assert warnings and "7/10" in warnings[0]
    # a zero target never warns
    assert splits.assign_component_splits("India", keyed([1] * 10), 1, 0.0, 0.0)[1] == []


def test_similarity_group_id_format():
    assert splits.similarity_group_id("India", 7) == "India:c000007"
    assert splits.similarity_group_id("Japan", 1234567) == "Japan:c1234567"
