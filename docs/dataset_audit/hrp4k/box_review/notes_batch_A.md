# Batch A visual review notes

- Sheets viewed: 91/91 (train_0001 to train_0091). Tiles: 1820, each id present once.
- Verdicts: ok 780, reject 135, unclear 905.
- Reject reasons: manhole 34, drain 28, kerb 22, stain 21, crack 12, patch 8, object 7, marking 2, plain 1.
- Unreviewed sheets: none.

Method: each sheet viewed once at sheet resolution (no zoom). Tiles under ~30 px (tiny pits on concrete slabs, thin dark slivers) were marked unclear rather than guessed. Verdicts are saved per group of 5-6 sheets.

Judgement calls:
- Rubber speed-hump joints (T00025, T00030, T00031): reject object.
- T00559-T00580 (except T00564, T00576): box on tyre/mud tracks in dust: reject stain.
- Gutter/kerb-edge boxes with a drain grate or kerb in frame: reject kerb/drain only when the box clearly framed the kerb, gutter or grate; otherwise unclear.
- Flush pale/yellow rectangular or oval plates, patches: reject patch/manhole only when clearly intact; ambiguous ones unclear.
- Cracks with visible missing material or a broken edge: ok; plain hairline cracks: reject crack.
- Puddle-filled holes with visible broken rim: ok; puddles on flat surface: unclear.
