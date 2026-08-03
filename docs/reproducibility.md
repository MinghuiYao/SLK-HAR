# Reproducibility status

The maintained artifact contains the SLK model, device-safe sparse masking,
tests, packaging, CI, and a smoke example. A complete experimental artifact
still requires dataset acquisition and checksums, preprocessing, subject split
files, training configurations, sparse schedule settings, seeds, checkpoints,
and generated result tables.

For sparse experiments, report initial density, prunable parameter policy,
prune and growth modes, update frequency, prune-rate schedule, final layer-wise
density, optimizer-state handling, dense and sparse parameter counts, and
measured latency/memory in addition to accuracy and macro-F1.
