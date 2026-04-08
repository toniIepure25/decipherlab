# Omniglot Integration Note

## Source
- Repository: `https://github.com/brendenlake/omniglot`
- Official archive: `images_background.zip` -> `https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip`
- Official archive: `images_evaluation.zip` -> `https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip`

## Local Storage
- Full download strategy: **full dataset**
- Downloaded archive size:
  - `images_background.zip`: `9464212` bytes
  - `images_evaluation.zip`: `6462886` bytes
- Extracted PNG size: `8182295` bytes
- Generated manifest size: `14568208` bytes
- Total local footprint under `data/raw/omniglot`: `43208095` bytes

This is well below the project’s `40 GB` maximum dataset budget.

## Preparation Strategy
- Extract both official archives under `data/raw/omniglot/extracted/`.
- Treat each image as one labeled glyph crop.
- Use `alphabet__character` as the transcription label.
- Use the alphabet name as `group_id`.
- Preserve the original Omniglot archive source (`images_background` vs `images_evaluation`) in per-record metadata.

## Split Strategy

Omniglot’s original background/evaluation split holds out whole character classes and therefore does not support same-label symbol recovery. To make the current fixed-vs-uncertainty protocol meaningful, we use a deterministic within-character split:

- train: `12` samples per character
- val: `4` samples per character
- test: `4` samples per character

This produces:

- `32460` total labeled crops
- `6492` test examples
- `19476` train examples
- `6492` val examples
- `1623` character classes
- `50` alphabet groups
- split seed: `23`

## Limitations

- Sequences are single-glyph crops, so sequence-level structural metrics are limited.
- Family labels matching the current decipherment hypothesis families are not available.
- The strongest evidence from this dataset is expected to remain symbol-level rather than semantic or family-level.
