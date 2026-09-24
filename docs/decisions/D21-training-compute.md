# D21. Training compute (settled in modelling Steps 4 and 7)

Status: Accepted for Level 0.

## Decision
Level 0 trains everything on the Mac CPU (4 threads): the Two-Tower takes about 45 s per epoch (2.3 minutes for 3 epochs) and DCN-V2 with MMoE about 17 s per epoch (1.1 minutes for 4 epochs); both are far under the one-hour target, so no spot training is needed for Level 0. Training runs as scripts with a config file (`configs/two_tower.json`, `configs/dcn.json`), each in its own process because PyTorch, FAISS and LightGBM cannot share a process on macOS.

## Why
Measured, not estimated (Steps 4 and 7). Peak memory 2.7 GB (DCN) and 1.7 GB (Two-Tower) fit in 8 GB.

## Alternatives considered
Colab or spot training: unnecessary at this size; revisit at Level 1 if the weekly data grows.
