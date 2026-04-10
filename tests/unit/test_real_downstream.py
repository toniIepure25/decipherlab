from __future__ import annotations

import numpy as np

from decipherlab.config import DecodingConfig, RealDownstreamConfig
from decipherlab.decoding.beam_search import (
    BeamDecodingResult,
    BigramTransitionModel,
    DecodedSequence,
    TrigramTransitionModel,
)
from decipherlab.models import GlyphCrop, SequenceExample, TranscriptionPosterior
from decipherlab.sequence.real_downstream import (
    build_supported_ngram_inventory,
    build_transcript_bank,
    downstream_supported_trigram_path_payload,
    downstream_transcript_bank_payload,
)


def _example(split: str, symbols: list[str], index: int) -> SequenceExample:
    glyphs = [
        GlyphCrop(position=position, image=np.zeros((4, 4), dtype=float), true_symbol=symbol)
        for position, symbol in enumerate(symbols)
    ]
    return SequenceExample(
        example_id=f"{split}_{index}",
        family=None,
        glyphs=glyphs,
        plaintext=None,
        observed_symbols=symbols,
        split=split,
        metadata={},
    )


def test_real_downstream_transcript_bank_payload_scores_same_length_bank():
    train_examples = [
        _example("train", ["ga", "gb", "gc", "gd"], 0),
        _example("train", ["ga", "gb", "ga", "gd"], 1),
    ]
    bank = build_transcript_bank(train_examples, RealDownstreamConfig(enabled=True))
    posterior = TranscriptionPosterior(
        candidate_ids=[["ga", "gb"], ["gb", "ga"], ["gc", "ga"], ["gd", "gc"]],
        log_probabilities=np.log(
            np.asarray(
                [
                    [0.9, 0.1],
                    [0.9, 0.1],
                    [0.85, 0.15],
                    [0.9, 0.1],
                ],
                dtype=float,
            )
        ),
    )
    bigram = BigramTransitionModel.fit([["ga", "gb", "gc", "gd"], ["ga", "gb", "ga", "gd"]], smoothing=0.1)
    trigram = TrigramTransitionModel.fit([["ga", "gb", "gc", "gd"], ["ga", "gb", "ga", "gd"]], smoothing=0.1)

    result = downstream_transcript_bank_payload(
        method="uncertainty_beam",
        posterior=posterior,
        truth=["ga", "gb", "gc", "gd"],
        transcript_bank=bank,
        real_downstream_config=RealDownstreamConfig(enabled=True, transcript_top_k=2),
        decoding_config=DecodingConfig(enabled=True, decoder_variants=["bigram_beam"], lm_weight=1.0),
        bigram_transition_model=bigram,
        trigram_transition_model=trigram,
    )

    assert result["real_downstream_supported"] is True
    assert result["real_downstream_bank_coverage"] == 1.0
    assert result["real_downstream_topk_recovery"] == 1.0
    assert result["real_downstream_best_transcript"] == ["ga", "gb", "gc", "gd"]


def test_real_downstream_supported_trigram_path_payload_tracks_supported_ngram_recovery():
    train_examples = [
        _example("train", ["ga", "gb", "gc", "gd"], 0),
        _example("train", ["gb", "gc", "gd", "ge"], 1),
    ]
    inventory = build_supported_ngram_inventory(
        train_examples,
        RealDownstreamConfig(enabled=True, task_name="train_supported_ngram_path", ngram_order=2),
    )
    decoded = BeamDecodingResult(
        sequences=[
            DecodedSequence(
                symbols=["ga", "gb", "gc", "gd"],
                total_score=0.0,
                visual_score=0.0,
                structural_score=0.0,
            ),
            DecodedSequence(
                symbols=["ga", "gb", "gc", "ge"],
                total_score=-1.0,
                visual_score=-1.0,
                structural_score=0.0,
            ),
        ],
        method="uncertainty_beam",
    )

    result = downstream_supported_trigram_path_payload(
        decoded=decoded,
        truth=["ga", "gb", "gc", "gd"],
        inventory=inventory,
        real_downstream_config=RealDownstreamConfig(
            enabled=True,
            task_name="train_supported_ngram_path",
            ngram_order=2,
        ),
    )

    assert result["real_downstream_supported"] is True
    assert result["real_downstream_bank_coverage"] == 1.0
    assert result["real_downstream_exact_match"] == 1.0
    assert result["real_downstream_topk_recovery"] == 1.0
    assert result["real_downstream_best_transcript"] == ["ga|gb", "gb|gc", "gc|gd"]
