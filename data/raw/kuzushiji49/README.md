# Kuzushiji-49 Integration Note

## Source
- Dataset: `Kuzushiji-49`
- OpenML ID: `41991`
- OpenML URL: `https://www.openml.org/d/41991`

## Local Storage
- Full downloaded cache size: `100113985` bytes
- Generated PNG size: `9288584` bytes
- Generated manifest size: `7451876` bytes
- Total local footprint under `data/raw/kuzushiji49`: `16740460` bytes

## Preparation Strategy
- Download the full Kuzushiji-49 corpus through OpenML and keep the cached archive intact.
- Preserve all 49 classes, but build a balanced crop manifest with deterministic per-class caps so the frozen multi-seed paper pack remains tractable.
- Treat each image as one single-position glyph sequence.
- Use the OpenML class label as `transcription`.
- Leave `family` and `group_id` unset because the source corpus does not provide decipherment-family or document-group structure compatible with the current downstream metrics.

## Subset Strategy
- Download mode: **full dataset download through OpenML cache**.
- Evaluation mode: **balanced capped subset with all classes preserved**.
- Train cap per class: `300`
- Validation cap per class: `75`
- Test cap per class: `75`
- Split seed: `23`

This produces:
- `22050` total labeled crops
- `3675` test examples
- `14700` train examples
- `3675` val examples
- `49` class labels

## Why This Strengthens The Paper
- Kuzushiji-49 is historically grounded and visually closer to manuscript conditions than scikit-learn digits.
- It complements Omniglot by adding a real historical character corpus with heavier class imbalance and different stroke statistics.
- It tests whether the symbol-level uncertainty-rescue effect survives on a corpus that is more manuscript-like without changing the core protocol.

## Limitations
- The current manifest is a balanced evaluation subset rather than a full-manifest sweep, chosen to keep the frozen multi-seed paper pack tractable while preserving all classes.
- Sequences are single-glyph crops, so grouped and downstream structural metrics remain limited.
- The strongest evidence from this dataset is still expected to remain symbol-level rather than semantic or family-level.
