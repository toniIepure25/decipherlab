# scikit-learn Digits Integration Note

## Source
- Loader: `sklearn.datasets.load_digits`
- Origin: `UCI Optical Recognition of Handwritten Digits`
- Additional network download: `0` bytes (dataset ships with the installed scikit-learn package).

## Local Storage
- Extracted PNG size: `220329` bytes
- Generated manifest size: `545885` bytes
- Total local footprint under `data/raw/sklearn_digits`: `766214` bytes

## Preparation Strategy
- Convert each 8x8 handwritten digit image into a single grayscale PNG crop.
- Treat each crop as one single-position sequence.
- Use the digit label as `transcription`.
- Leave `family` and `group_id` unset because the source corpus does not provide decipherment-family or document-group structure.

## Split Strategy
- Deterministic per-class split using a fixed random seed.
- Train cap per class: `100`
- Validation cap per class: `30`
- Test policy: `remainder`
- Split seed: `23`

This produces:
- `1797` total labeled crops
- `497` test examples
- `1000` train examples
- `300` val examples
- `10` digit classes

## Limitations
- This is a real handwritten-symbol corpus, but it is low-resolution and far less script-like than Omniglot or Kuzushiji-49.
- Sequences are single-glyph crops, so downstream structural and grouped metrics remain limited.
- The strongest evidence from this dataset is expected to remain symbol-level rather than decipherment-level.
