"""Evaluation of trained detector weights.

Responsibility: compute detection metrics for a weights file on a dataset split.

Intended contract:
- Evaluation uses standard Ultralytics validation settings, with no confidence
  pre-filter.
- The ``val`` split is the default.
- Evaluating the held-out test split requires an explicit flag. Each such
  evaluation appends a record (timestamp, weights hash, dataset manifest hash,
  metrics) to an append-only ledger file, so the number of test-set evaluations
  is auditable.

Not implemented yet.
"""
