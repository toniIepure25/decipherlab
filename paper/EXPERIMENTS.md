# Experiments

## Protocol

- We evaluate the existing four-condition protocol: fixed vs uncertainty-aware inference crossed with heuristic vs calibrated posterior generation.
- Omniglot and Kuzushiji-49 use ambiguity levels `[0.0, 0.45]` and a two-seed sweep `{23, 29}`.
- scikit-learn digits uses ambiguity levels `[0.0, 0.15, 0.3, 0.45]` and a three-seed sweep `{23, 29, 31}` from the earlier completed run.
- Evaluation top-k: `5`.
- Bootstrap procedure: `500` trials at confidence `0.95`.
- Each dataset is run independently under the same reporting schema, then synthesized at the results-pack level.

## Dataset 1: Omniglot

- Run directory: `outputs/runs/20260407T150327Z_omniglot_paper_pack_evaluation`
- Manifest: `data/raw/omniglot/manifest.yaml`
- Sequences / records: `32,460`
- Split composition: train=`19,476`, val=`6,492`, test=`6,492`
- Symbol classes: `1,623`
- Group structure: `50` alphabet groups
- Label coverage: `1.0` in every split

## Dataset 2: scikit-learn Digits

- Run directory: `outputs/runs/20260407T191829Z_sklearn_digits_paper_pack_evaluation`
- Manifest: `data/raw/sklearn_digits/manifest.yaml`
- Sequences / records: `1,797`
- Split composition: train=`1,000`, val=`300`, test=`497`
- Symbol classes: `10`
- Group structure: none
- Label coverage: `1.0` in every split

## Dataset 3: Kuzushiji-49

- Run directory: `outputs/runs/20260407T201933Z_kuzushiji49_balanced_subset_paper_pack_evaluation`
- Manifest: `data/raw/kuzushiji49/manifest_balanced_49x300_75_75.yaml`
- Sequences / records: `22,050`
- Split composition: train=`14,700`, val=`3,675`, test=`3,675`
- Symbol classes: `49`
- Group structure: none in the current manifest
- Label coverage: `1.0` in every split
- Additional note: the full dataset was downloaded through OpenML, but the experiment manifest uses a deterministic balanced cap to preserve all classes while keeping the frozen multi-seed run tractable

## Reporting

- Primary symbol-level metrics: top-1, top-k, NLL, and ECE.
- Downstream family metrics are reported only when labels permit them; none of the three datasets provides strong family supervision for the current hypothesis families.
- Failure summaries are preserved in the main evidence pack rather than omitted as secondary artifacts.
- Cross-dataset synthesis is reported through `outputs/cross_dataset_summary.*`, `outputs/cross_dataset_tables_with_ci.csv`, `outputs/cross_dataset_failure_summary.csv`, and `outputs/cross_dataset_effects_plot.png`.
