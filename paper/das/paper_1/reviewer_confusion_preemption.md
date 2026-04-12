# Reviewer Confusion Preemption

## 1. What is fully real

- The grouped Historical Newspapers benchmark is real grouped data with OCR-derived labels that were later audited and gold-style checked.
- The ScaDS.AI grouped benchmark is real grouped data with public word-level annotations.
- The replicated real grouped result is therefore genuinely real at the grouped-recognition level.

## 2. What is synthetic-from-real

- The Omniglot, scikit-learn Digits, and Kuzushiji-49 sequence and downstream tasks are synthetic-from-real in structure.
- They are used to test higher-level grouped and downstream behavior under controlled ambiguity, not to claim fully real document-level structure.

## 3. What replicates

- Real grouped top-k rescue replicates across Historical Newspapers and ScaDS.AI.
- Symbol top-k rescue also transfers across both real grouped corpora.

## 4. What is selective

- Exact grouped recovery does not replicate cleanly across both real grouped corpora.
- Exact real downstream gains remain selective even after the redesigned downstream task improves coverage substantially.
- Conformal helps in some support regimes, but not universally.

## 5. What the paper's actual discovery is

- The paper's discovery is not a new decoder.
- The paper shows that preserved uncertainty can produce a replicated real grouped signal, but that downstream propagation depends on measurable support conditions rather than following automatically from grouped rescue.

## 6. Why the paper is still worthwhile despite mixed exact-downstream results

- The real grouped result is replicated across two corpora.
- The Historical Newspapers benchmark is trust-hardened enough for a bounded claim.
- The redesigned downstream task removes the strongest trivial coverage objection.
- The support-aware propagation analysis explains why downstream gains remain mixed, which is a stronger scientific contribution than simply reporting another negative or selective downstream result.
