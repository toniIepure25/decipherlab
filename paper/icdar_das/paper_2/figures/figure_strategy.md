# Figure Strategy for Paper 2

## Figure 1: System pipeline

Purpose:

- show where the adaptive controller sits in the existing pipeline
- make paper_2 feel like a document-analysis system paper

Content:

- glyph crops
- posterior / confusion network
- support feature extraction
- adaptive controller
- decoder choice
- grouped / downstream outputs

## Figure 2: Controller logic

Purpose:

- make the method interpretable in one glance

Content:

- support feature inputs
- low-entropy compact regime -> conformal path
- high-entropy diffuse regime -> raw uncertainty path
- beam-width adaptation

This figure should feel like a policy diagram, not a neural architecture diagram.

## Figure 3: Results

Purpose:

- show whether the controller helps on the real grouped and real downstream tasks

Content:

- grouped top-k delta vs fixed
- downstream exact delta vs fixed
- adaptive vs strongest fixed baseline

## Figure 4: Failure / regime figure

Purpose:

- show which failure regimes are reduced and which remain

Content:

- controller selection rates by regime
- avoided conformal-over-pruning cases
- grouped rescue without downstream recovery cases that remain

## Main design rule

Paper 2 figures should look like system and method figures, not like a continuation of the propagation-analysis dashboard from paper_1.
