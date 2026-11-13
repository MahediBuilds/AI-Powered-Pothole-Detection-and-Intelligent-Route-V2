# Batch C review notes

- Sheets viewed: 90 / 90 (train_0183 to train_0263, valid_0001 to valid_0009). Every sheet was opened and judged visually; none skipped.
- Tiles: 1799. Verdicts: ok 514, reject 80, unclear 1205.
- Reject reasons: manhole 39, stain 23 (tyre marks on dust/sand), drain 6, marking 5, patch 3, kerb 2, crack 1, object 1 (bird).
- Method: conservative. Sheet crops are low resolution, and many boxes are tiny dark pits or pale patches on concrete/asphalt. `unclear` was used whenever a pothole could not be told apart from a stain, flush patch, joint or chip. `ok` was given only for visible missing material or a clear hole. `reject` only for clearly identifiable covers, tyre tracks, markings and similar.
- Pale oval fills on asphalt (possible flush cold-mix patches) were mostly left `unclear`.
- Many tiles from the same source image appear in runs of 3 to 10 (same scene, different box); verdicts within a run are usually consistent but were assigned per tile.
