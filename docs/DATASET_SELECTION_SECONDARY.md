# Secondary Dataset Selection

## Goal

Select one second real handwritten-symbol dataset that complements Omniglot under the frozen DecipherLab paper pack.

The selection criteria were:

- genuine relevance to crop-level visual ambiguity
- different visual statistics from Omniglot
- compatibility with the existing manifest workflow
- minimal architecture churn
- clear storage accounting

## Candidates Considered

| Dataset | Why It Was Considered | Strengths | Weaknesses |
| --- | --- | --- | --- |
| Kuzushiji-49 | Best historical-script complement to Omniglot | Historical cursive characters, strong symbol labels, very different visual texture from Omniglot | The official host timed out repeatedly from this execution environment, so practical integration was blocked even before adapter work |
| EMNIST ByClass | Large real handwritten character corpus with strong labels | Much larger than the bundled digits corpus, easy symbol-level framing | Not locally available in this environment, no natural grouping, less aligned with manuscript-style framing |
| scikit-learn Digits | Already available locally and visually distinct from Omniglot | Real handwritten symbols, zero network download, deterministic and clean integration into the frozen manifest workflow | Small, low-resolution, and less script-like than Kuzushiji-49 |

## Chosen Dataset

**scikit-learn Digits** was selected for this phase.

It won as the strongest practical complement to Omniglot under the actual execution constraints. Kuzushiji-49 remained the preferred scientific option, but repeated download attempts to the official host timed out in this environment. Rather than stall the cross-dataset phase or broaden the system, the project uses the bundled digits corpus to test whether the symbol-level uncertainty effect survives on a second real handwritten-symbol dataset with very different image statistics.

## Why It Complements Omniglot

- Omniglot is multi-script and high-variation; scikit-learn digits is low-resolution, single-domain, and numerically constrained.
- Omniglot uses `1,623` classes across `50` alphabets; scikit-learn digits uses `10` classes with much denser per-class support.
- The different visual scale and class structure let us test whether the symbol-level rescue effect is specific to Omniglot-like script inventories or persists in a simpler handwritten-symbol setting.

## Dataset Accounting

- Dataset name: `scikit-learn Digits`
- Source loader: `sklearn.datasets.load_digits`
- Upstream origin: `UCI Optical Recognition of Handwritten Digits`
- Approximate full size estimate: bundled with the installed scikit-learn package; no separate external archive was downloaded
- Downloaded size: `0` bytes from the network in this environment
- Local extracted footprint under `data/raw/sklearn_digits/`: `766,214` bytes
- Download mode: **bundled/local**
- Experimental subset strategy: **none**; the full bundled corpus is used

## Environment Limitation

Kuzushiji-49 was not rejected on scientific grounds. It was rejected on practical execution grounds for this run:

- repeated shell-level attempts to reach `codh.rois.ac.jp` timed out
- the frozen paper-pack phase depends on having a concrete second dataset in the workspace
- switching to a bundled real dataset preserved the evaluation protocol and allowed a full cross-dataset run

## Known Limitations

- scikit-learn digits is a weaker manuscript proxy than Kuzushiji-49.
- It provides no natural document grouping or decipherment-family labels.
- The resulting cross-dataset synthesis remains strongest at the symbol level rather than the downstream structural level.
