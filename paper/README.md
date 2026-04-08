# Paper Workspace

This directory holds workshop-paper drafts grounded in the current DecipherLab protocol.

For the final venue-agnostic handoff package, see [submission/README.md](/home/tonystark/Desktop/decipher/submission/README.md).

## Stable Drafts

- `TITLE_ABSTRACT_INTRO.md`
- `METHODS.md`
- `CONCLUSION.md`
- `CLAIMS_AUDIT.md`

These files describe the current system and supported claims without depending on one specific run directory.

## Run-Refreshed Drafts

- `EXPERIMENTS.md`
- `RESULTS.md`
- `LIMITATIONS.md`

Refresh them from a completed results pack with:

```bash
python3 scripts/assemble_paper_from_results.py --run-dir outputs/runs/<run_dir> --paper-dir paper
```

For a full manifest-backed external validation run plus manuscript refresh:

```bash
python3 scripts/run_real_manifest_paper_pack.py --config configs/experiments/real_manifest_large_paper_pack.yaml --paper-dir paper
```
