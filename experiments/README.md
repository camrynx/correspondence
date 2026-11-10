# Experiment drivers

Batch scripts used for the additional experiments. They resume from done-markers in
`state/` and write logs to `logs/`. GPU stages wait for free GPU memory (`GATE_MB`).

`compute_data/` reruns the English pipeline on 10 min up to the full training split
(DUSTED, features, alignments, cAE and NCE cAE, AP and ABX), one sandbox per size, and
logs time and memory for every stage. It also adapts and evaluates SpidR-Adapt on the
same audio. It needs the DUSTED pipeline (`UTD_DUSTED_ROOT`, default
`../UTD_DUSTED_Clean`) and SpidR-Adapt (`SPIDR_SRC`, default `../spidr-adapt/src`).
`results.py` and `plot.py` summarise the runs.

`ccae_runs/` has the runs for the NCE cAE tables and some later reruns. `status.py`
shows progress and `check_alignments.py` checks that alignments match their features.

`PY` sets the Python interpreter. These scripts were moved over from an older version
of the repo and have not been rerun since.
