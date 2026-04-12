from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = 0
    seed_sweep: list[int] = Field(default_factory=list)
    output_root: Path = Path("outputs/runs")
    notes: str | None = None


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["synthetic", "manifest"] = "synthetic"
    manifest_path: Path | None = None
    manifest_format: Literal["synthetic_npz", "glyph_crop"] = "glyph_crop"
    generate_if_missing: bool = True
    train_split: str = "train"
    val_split: str = "val"
    evaluation_split: str = "test"
    min_sequences_per_split_warning: int = 2
    min_symbol_instances_per_train_class_warning: int = 2
    min_family_instances_per_split_warning: int = 2


class SyntheticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    families: list[Literal["monoalphabetic", "homophonic", "transposition", "pseudo_text"]]
    samples_per_family: int = 8
    sequence_length: int = 48
    alphabet_size: int = 18
    homophonic_extra_symbols: int = 8
    transposition_block_size: int = 4
    noise_std: float = 0.2
    allograph_jitter: float = 1.25
    image_size: int = 24
    train_fraction: float = Field(default=0.6, gt=0.0, lt=1.0)
    val_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_split_fractions(self) -> "SyntheticConfig":
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError("train_fraction + val_fraction must be < 1.0.")
        return self


class VisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_downsample: int = 10
    estimate_clusters: bool = True
    min_clusters: int = 4
    max_clusters: int = 32


class PosteriorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["cluster_distance", "calibrated_classifier"] = "calibrated_classifier"
    top_k: int = 4
    temperature: float = 1.5
    floor_probability: float = Field(default=1.0e-6, gt=0.0)
    embedding_dim: int = 32
    use_label_supervision: bool = True
    calibration_grid: list[float] = Field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    )


class TriageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repeat_ngram_sizes: list[int] = Field(default_factory=lambda: [2, 3])
    shuffled_null_trials: int = 8


class HypothesisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    families: list[
        Literal[
            "unknown_script",
            "monoalphabetic",
            "homophonic",
            "transposition_heuristic",
            "pseudo_text_null",
        ]
    ]


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    ambiguity_levels: list[float] = Field(default_factory=lambda: [0.0, 0.15, 0.3, 0.45])
    noise_levels: list[float] | None = None
    top_k: int = 3
    comparison_strategies: list[Literal["cluster_distance", "calibrated_classifier"]] = Field(
        default_factory=lambda: ["cluster_distance", "calibrated_classifier"]
    )
    overdiffuse_entropy_ratio: float = Field(default=0.8, ge=0.0, le=1.5)
    bootstrap_trials: int = Field(default=250, ge=0)
    bootstrap_confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    bootstrap_seed: int | None = None


class SequenceBenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    task_name: Literal[
        "real_glyph_markov_sequences",
        "real_glyph_process_family_sequences",
        "real_grouped_manifest_sequences",
    ] = "real_glyph_markov_sequences"
    selected_symbol_count: int = Field(default=8, ge=2)
    min_instances_per_symbol: int = Field(default=12, ge=2)
    train_sequences: int = Field(default=96, ge=1)
    val_sequences: int = Field(default=32, ge=1)
    test_sequences: int = Field(default=32, ge=1)
    sequence_length: int = Field(default=12, ge=2)
    group_count: int = Field(default=3, ge=1)
    self_transition_bias: float = Field(default=3.0, gt=0.0)
    within_group_bias: float = Field(default=1.5, gt=0.0)
    cross_group_bias: float = Field(default=0.35, gt=0.0)
    transition_noise: float = Field(default=0.05, ge=0.0)
    sample_with_replacement: bool = True
    process_families: list[
        Literal["sticky_markov", "alternating_markov", "motif_repeat"]
    ] = Field(
        default_factory=lambda: ["sticky_markov", "alternating_markov", "motif_repeat"]
    )
    motif_length: int = Field(default=3, ge=2)
    motif_noise_probability: float = Field(default=0.1, ge=0.0, le=1.0)
    minimum_real_sequence_length: int = Field(default=4, ge=2)
    maximum_real_sequence_length: int | None = Field(default=None, ge=2)


class StructuredUncertaintyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    representation: Literal["confusion_network"] = "confusion_network"
    max_candidates_per_position: int = Field(default=6, ge=1)
    cumulative_probability_mass: float = Field(default=0.95, gt=0.0, le=1.0)
    min_probability: float = Field(default=1.0e-5, gt=0.0)
    include_top1_fallback: bool = True


class DecodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    strategy: Literal["bigram_beam"] = "bigram_beam"
    decoder_variants: list[Literal["bigram_beam", "trigram_beam", "crf_viterbi"]] = Field(
        default_factory=lambda: ["bigram_beam"]
    )
    beam_width: int = Field(default=8, ge=1)
    lm_weight: float = Field(default=1.0, ge=0.0)
    trigram_lm_weight: float | None = Field(default=None, ge=0.0)
    transition_smoothing: float = Field(default=0.1, gt=0.0)
    top_k_sequences: int = Field(default=5, ge=1)
    length_normalize: bool = True


class AdaptiveDecodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    policy: Literal[
        "support_aware_rule",
        "support_aware_learned_gate",
        "support_aware_constrained_gate",
        "support_aware_profiled_gate",
        "support_aware_profile_selector",
    ] = "support_aware_rule"
    operating_profile: Literal["rescue_first", "shortlist_first"] = "rescue_first"
    review_budget_k: Literal[2, 3, 5] = 3
    conformal_entropy_threshold: float = Field(default=0.5, ge=0.0)
    raw_entropy_threshold: float = Field(default=1.15, ge=0.0)
    conformal_set_size_threshold: float = Field(default=2.0, ge=1.0)
    minimum_set_size_for_conformal: float = Field(default=1.1, ge=1.0)
    raw_set_size_threshold: float = Field(default=3.5, ge=1.0)
    low_support_length_count_threshold: int = Field(default=4, ge=0)
    narrow_beam_width: int = Field(default=4, ge=1)
    wide_beam_width: int = Field(default=12, ge=1)
    prefer_conformal_for_calibrated: bool = True
    learned_gate_learning_rate: float = Field(default=0.2, gt=0.0)
    learned_gate_steps: int = Field(default=2000, ge=100)
    learned_gate_l2_penalty: float = Field(default=1.0e-2, ge=0.0)
    learned_gate_decision_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    learned_gate_prediction_set_penalty: float = Field(default=0.05, ge=0.0)
    constrained_min_utility_margin: float = Field(default=0.02, ge=0.0)
    constrained_max_grouped_topk_drop: float = Field(default=0.02, ge=0.0)
    constrained_max_grouped_exact_drop: float = Field(default=0.0, ge=0.0)
    constrained_min_downstream_exact_gain: float = Field(default=0.0)
    constrained_max_wide_set_size_inflation: float = Field(default=0.75, ge=0.0)
    constrained_min_wide_grouped_topk_gain: float = Field(default=0.02, ge=0.0)
    constrained_conformal_threshold: float = Field(default=0.6, gt=0.0, lt=1.0)
    constrained_wide_threshold: float = Field(default=0.55, gt=0.0, lt=1.0)
    shortlist_utility_weight: float = Field(default=0.75, ge=0.0)
    tight_review_budget_threshold: int = Field(default=3, ge=1)
    budget_threshold_delta: float = Field(default=0.08, ge=0.0, le=0.5)
    enable_defer_to_human: bool = True
    defer_entropy_threshold: float = Field(default=1.2, ge=0.0)
    defer_set_size_threshold: float = Field(default=3.0, ge=1.0)
    defer_length_support_threshold: int = Field(default=2, ge=0)
    defer_min_fragile_signals: int = Field(default=2, ge=1)
    selector_learning_rate: float = Field(default=0.2, gt=0.0)
    selector_steps: int = Field(default=2000, ge=100)
    selector_l2_penalty: float = Field(default=1.0e-2, ge=0.0)
    selector_decision_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    selector_min_effort_margin: float = Field(default=0.01, ge=0.0)
    selector_max_grouped_topk_drop: float = Field(default=0.04, ge=0.0)
    selector_min_downstream_exact_gain: float = Field(default=-0.01)
    selector_enable_direct_defer: bool = True
    selector_defer_margin: float = Field(default=0.08, ge=0.0, le=0.5)


class RiskControlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["split_conformal"] = "split_conformal"
    alpha: float = Field(default=0.1, gt=0.0, lt=1.0)
    min_set_size: int = Field(default=1, ge=1)
    max_set_size: int | None = Field(default=None, ge=1)
    include_top1_fallback: bool = True


class RealDownstreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    task_name: Literal[
        "train_transcript_bank",
        "train_supported_ngram_path",
        "train_supported_trigram_path",
    ] = "train_transcript_bank"
    transcript_top_k: int = Field(default=5, ge=1)
    min_frequency: int = Field(default=1, ge=1)
    exact_length_only: bool = True
    missing_log_probability: float = Field(default=1.0e-8, gt=0.0)
    ngram_order: int = Field(default=3, ge=2)
    min_supported_ngrams: int = Field(default=1, ge=1)


class DecipherLabConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig
    dataset: DatasetConfig
    synthetic: SyntheticConfig
    vision: VisionConfig
    posterior: PosteriorConfig
    triage: TriageConfig
    hypotheses: HypothesisConfig
    evaluation: EvaluationConfig
    sequence_benchmark: SequenceBenchmarkConfig = Field(default_factory=SequenceBenchmarkConfig)
    structured_uncertainty: StructuredUncertaintyConfig = Field(default_factory=StructuredUncertaintyConfig)
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)
    adaptive_decoding: AdaptiveDecodingConfig = Field(default_factory=AdaptiveDecodingConfig)
    risk_control: RiskControlConfig = Field(default_factory=RiskControlConfig)
    real_downstream: RealDownstreamConfig = Field(default_factory=RealDownstreamConfig)


def load_config(path: str | Path) -> DecipherLabConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return DecipherLabConfig.model_validate(payload)


def dump_config(config: DecipherLabConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config.model_dump(mode="json"),
            handle,
            sort_keys=False,
            default_flow_style=False,
        )
