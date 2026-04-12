# Practical DAS Use Case

## Who uses the system

The most natural user is an archive digitization or historical-document transcription team working with grouped word crops that still contain uncertainty after the initial recognition stage.

## Workflow

1. a recognizer produces confusion-network style uncertainty over grouped token crops
2. the adaptive controller inspects support features such as entropy, set size, and train-support density
3. it chooses whether to preserve alternatives with raw uncertainty decoding or prune with conformal decoding
4. the system returns grouped candidates and, where available, downstream structural candidates
5. a human operator reviews the shortlist rather than a single brittle transcript

## Why the adaptive controller matters

A static decoder is awkward in practice because archive workloads are heterogeneous:

- some items are diffuse and need preserved alternatives
- some items are compact and benefit from pruning
- some items have enough support to justify wider search
- some items are fragile enough that aggressive pruning is risky

The controller tries to adapt that behavior per item while staying auditable.

## Why grouped top-k rescue still matters operationally

Even when exact downstream recovery is mixed, grouped top-k rescue has practical value:

- it improves candidate ranking for archivists
- it reduces premature collapse to one wrong transcript
- it supports verification and correction workflows
- it makes the system more useful as transcription assistance rather than only as a one-shot recognizer

## Current honest boundary

The current controller does not yet produce a universal exact-recovery gain. Its present value is operational rather than grandiose: it is a small decision layer that tries to preserve the right alternatives in the right regime.
