# Assessing Resource Availability for Business Process Simulation Calibration

This is the code for our paper *Assessing Resource Availability for Business Process
Simulation Calibration*.

## Code layout

`resource/scripts/main.py` runs the stages below in order.

- `preparation.py` loads the synthetic configs and simulates the baseline, P and P_TRUE
- `estimation.py` builds the row and group features and trains the Q+ scorer with CatBoost
- `diagnosis.py` sets the thresholds the decrease rule needs
- `calibration.py` builds P_cal, re-simulates it and scores the behavioural distances
- `evaluation.py` aggregates the per-seed metrics
- `reports.py` writes the series behind the figures

The rest of `resource/scripts/`:

- `_core.py` holds the constants, the helpers and the shared computations
- `simulation.py` generates an event log from a scenario config and an availability surface
- `sensitivity.py` runs the chain once per step of either sensitivity analysis

## Setup

Clone and install:

```bash
git clone --recurse-submodules <repo-url>
git submodule update --init --recursive

cd resource
pip install -r requirements.txt
```

## Running the evaluations

```bash
cd resource
python scripts/main.py --synthetic --logs data/synthetic/logs   # into results/synthetic
python scripts/main.py --real                                   # into results/real
```

Both run every stage in order. Each stage can also be run on its own:

```bash
RUN=results/synthetic
export SYNTH_LOG_STORE=data/synthetic/logs
python scripts/preparation.py --run-dir $RUN --synthetic
python scripts/estimation.py  --run-dir $RUN --synthetic
python scripts/diagnosis.py   --run-dir $RUN --synthetic
python scripts/calibration.py --run-dir $RUN --synthetic
python scripts/evaluation.py  --run-dir $RUN
python scripts/reports.py     --run-dir $RUN
```

## Input data and the supplementary material

The input data is under [`resource/data`](resource/data).

[`supplementary/supplementary_material.pdf`](supplementary/supplementary_material.pdf) is
the supplementary material for our paper, including the two sensitivity analyses.

```bash
python scripts/sensitivity.py --resources   # into results/sensitivity_deviating_resources
python scripts/sensitivity.py --workload    # into results/sensitivity_arrival_rate
```
