# Greenhouse Predictive Control Roadmap

## 1. Core Judgment

### 1.1 Can prediction quality and control quality be optimized together?

Yes, but not automatically.

They are related, not identical objectives:

- A model with strong average forecasting metrics can still be weak for control.
- A model that supports good control may sacrifice some average forecast accuracy.
- The key difference is that control needs the model to be locally correct in the directions induced by actions and disturbances.

In greenhouse control, a forecasting model is useful for MPC only if it is:

- action-conditioned
- stable over the full prediction horizon
- sensitive to control inputs in the correct direction
- robust to forecast error and rollout error
- consistent under closed-loop rollout

This means:

- `best MAE/R2 model` is not always the `best MPC model`
- `best control model` is not always the `best offline predictor`

### 1.2 Should prediction and control share one model?

There are three valid designs:

1. One shared model for both forecasting and control.
2. One shared backbone with separate heads or separate training objectives.
3. Two different models: one optimized for forecasting, one optimized for control.

For this project, the most practical recommendation is:

- keep a strong forecasting benchmark model
- build a control-oriented dynamics model separately or from the same backbone
- compare them in both offline prediction and closed-loop control

### 1.3 Recommended conclusion for the thesis

Do not assume forecasting and control are the same problem.

A stronger thesis statement is:

> In greenhouse MPC, the model with the best offline forecasting metrics is not necessarily the model that yields the best closed-loop control. This work studies control-oriented forecasting and closed-loop evaluation under future weather and future control information.


## 2. Project Positioning

This project should not be framed as:

- "a better Transformer for greenhouse forecasting"

It should be framed as:

- "control-oriented multi-step forecasting for greenhouse MPC"
- "closed-loop benchmark of data-driven predictive control under future weather and action information"

This positioning is stronger because it focuses on:

- a real control problem
- action-conditioned forecasting
- strict closed-loop evaluation
- reproducible benchmarks


## 3. Main Research Goal

Build a greenhouse prediction-and-control framework that:

- uses historical indoor data
- uses future weather / exogenous forecasts
- uses future control candidates or planned control sequences
- predicts future indoor greenhouse variables over a multi-step horizon
- supports closed-loop MPC / DPC
- outperforms baseline forecasting and baseline control methods on relevant metrics


## 4. Research Questions

### RQ1

Which model class is best for greenhouse control-oriented multi-step forecasting:

- linear models
- RNN-based models
- Transformer-based models
- hybrid residual models
- graph-structured models

### RQ2

Does stronger offline forecasting imply stronger closed-loop control?

### RQ3

How much do future weather forecasts and future action information improve multi-step greenhouse prediction?

### RQ4

Can hybrid or uncertainty-aware models improve robustness under forecast error and model mismatch?

### RQ5

Can the proposed framework generalize better than current baselines across datasets or greenhouse settings?


## 5. Minimum Thesis Scope

This is the minimum scope that can support a strong master's thesis.

### 5.1 Datasets

- Main dataset: AGC 2019
- Secondary dataset: current strawberry dataset

Why:

- AGC 2019 is richer, more realistic, and more suitable for MPC
- Strawberry is useful as a low-data, weak-actuation stress test

### 5.2 Prediction targets

Start with:

- indoor temperature
- indoor relative humidity or humidity deficit
- indoor CO2
- optionally PAR / light

### 5.3 Control variables

Prefer setpoint-style controls where available:

- heating temperature setpoint
- ventilation temperature setpoint
- CO2 setpoint
- humidity-related setpoint
- lighting / screen / irrigation controls

For strawberry, keep the current simplified control space, but explicitly state it is a reduced testbed.

### 5.4 Forecasting baselines

At minimum include:

- persistence / naive
- linear model or ARX / state-space baseline
- GRU or LSTM
- DLinear or NLinear
- SegRNN
- current Transformer-hybrid

### 5.5 Control baselines

At minimum include:

- rule-based or heuristic baseline
- classical identified-model MPC
- current DPC / differentiable MPC
- SAC baseline

### 5.6 Evaluation

Offline forecasting:

- MAE
- RMSE
- R2
- horizon-wise metrics
- final-step metrics

Closed-loop control:

- tracking MAE
- energy or control cost
- CO2 / water / electricity proxy cost when available
- action total variation
- constraint violation rate
- robustness under forecast disturbance


## 6. Stronger Method Direction

The recommended main method is not "a bigger Transformer".

The recommended main method is:

## Hybrid Control-Oriented Forecasting Model

Suggested design:

- history branch for past indoor observations
- exogenous branch for future weather forecasts
- action branch for future control inputs
- residual output anchored to the latest state
- uncertainty head for probabilistic forecasts

Good candidate variants:

1. GRU / SegRNN + future exogenous/action conditioning
2. Transformer-hybrid with explicit past/future separation
3. hybrid residual model: simple physics prior + learned residual
4. graph-structured model if multi-variable coupling is emphasized

### Key training ideas

- direct multi-step prediction
- horizon-weighted loss
- variable-weighted loss
- delta / derivative consistency loss
- rollout consistency loss
- action sensitivity regularization


## 7. Proposed Contribution Structure

### Contribution 1

A control-oriented greenhouse forecasting benchmark with:

- past observations
- future weather
- future actions
- strict leak-free closed-loop evaluation

### Contribution 2

A multi-step dynamics model tailored for MPC, rather than pure offline forecasting.

### Contribution 3

A comparative study showing that:

- the best offline predictor is not always the best closed-loop controller model
- simple models can outperform Transformers on small greenhouse datasets
- hybrid or uncertainty-aware models are more robust for control

### Contribution 4

Optional stronger contribution:

- cross-dataset transfer between AGC and strawberry
- or forecast-error-aware robust MPC


## 8. What Counts as Innovation

### Weak innovation by itself

- "multi-parameter coupling"
- "using Transformer"
- "using SAC as a baseline"
- "doing MPC on greenhouse data"

These are not enough alone.

### Medium-strength innovation

- action-conditioned multi-step forecasting
- future weather plus future action conditioning
- strict closed-loop, no-leak evaluation
- joint forecasting-control benchmark on public greenhouse datasets

### Stronger innovation

- control-oriented objective design
- hybrid physics + learned residual model
- uncertainty-aware MPC
- cross-dataset transfer / adaptation
- graph or structured coupling model with closed-loop gains


## 9. Detailed Work Plan

### Phase 0. Lock the problem definition

Deliverables:

- final thesis title direction
- chosen primary dataset
- chosen variables and control space
- final benchmark protocol

Decisions:

- make AGC 2019 the main benchmark
- keep strawberry as secondary stress-test benchmark
- define one primary control task and one secondary task

### Phase 1. Build a clean benchmark pipeline

Tasks:

- unify data schema across datasets
- define `x_past`, `w_future`, `u_future`, `y_future`
- ensure no future leakage
- define train / validation / test splits
- implement strict closed-loop simulator

Exit criteria:

- one command reproduces all prediction benchmarks
- one command reproduces all control benchmarks

### Phase 2. Establish strong baselines

Tasks:

- naive / persistence
- linear or state-space model
- GRU / LSTM
- DLinear / NLinear
- SegRNN
- current Transformer-hybrid
- classical MPC baseline
- current DPC
- SAC baseline

Exit criteria:

- complete offline prediction table
- complete closed-loop control table

### Phase 3. Analyze the forecasting-control gap

Tasks:

- rank models by offline metrics
- rank models by closed-loop metrics
- compare ranking mismatch
- identify which error patterns matter most for control

Expected thesis value:

- this can become one of the core empirical findings

### Phase 4. Implement the main proposed method

Preferred order:

1. hybrid residual model
2. uncertainty-aware extension
3. graph-coupled extension if still needed

Tasks:

- add explicit past/future branches
- add residual anchoring
- add uncertainty or structured coupling
- add control-oriented loss

Exit criteria:

- the proposed model wins on at least one important closed-loop dimension

### Phase 5. Robustness and ablations

Tasks:

- remove future action input
- remove future weather input
- remove residual anchoring
- remove uncertainty head
- inject weather forecast noise
- vary horizon length
- vary control horizon

Exit criteria:

- clear explanation of why the proposed model works

### Phase 6. Thesis-ready packaging

Tasks:

- final figures
- final tables
- chapter structure
- reproducibility scripts
- limitations and future work


## 10. Thesis Chapter Outline

### Chapter 1. Introduction

- background
- motivation
- research gap
- contributions

### Chapter 2. Literature Review

- greenhouse predictive control
- MPC categories
- data-driven greenhouse forecasting
- Transformer and non-Transformer time-series models
- AGC and public greenhouse datasets

### Chapter 3. Problem Formulation

- datasets
- variables
- forecasting task
- control task
- metrics

### Chapter 4. Methods

- benchmark models
- proposed model
- MPC / DPC controller
- closed-loop simulation protocol

### Chapter 5. Experiments

- offline forecasting
- closed-loop control
- ablation
- robustness
- transfer or generalization

### Chapter 6. Discussion

- forecasting vs control gap
- limitations
- practical relevance

### Chapter 7. Conclusion

- summary
- contributions
- future work


## 11. What Is Enough for a Strong Master's Thesis

The following is enough:

- one strong main dataset plus one secondary dataset
- 5 to 6 meaningful forecasting baselines
- 3 to 4 control baselines
- one clear proposed method
- strict offline and closed-loop evaluation
- one strong empirical conclusion about forecasting vs control

If this is executed well, the thesis is strong even without a real greenhouse deployment.


## 12. What Must Be Added for a Q3-Level Paper

To move from thesis-level to Q3-level, add at least three of the following:

- AGC 2019 as the main benchmark
- stronger baselines beyond current Transformer
- one genuinely novel method component
- robustness under forecast uncertainty
- transfer across datasets or greenhouse settings
- economic or resource-aware control metrics
- extensive ablation and statistical testing
- public benchmark code release

The strongest Q3 paper angle is:

> A control-oriented greenhouse forecasting benchmark and hybrid predictive control model under future weather and action information.


## 13. Immediate Next Actions

### Immediate priority A

Replace "model-centric" thinking with "benchmark-centric" thinking.

### Immediate priority B

Move the main experiment set to AGC 2019.

### Immediate priority C

Benchmark simple strong models before any new Transformer changes.

### Immediate priority D

Implement one stronger proposed model:

- hybrid residual GRU/SegRNN
- or uncertainty-aware action-conditioned Transformer-hybrid


## 14. Decision Rule for Future Iterations

For every new model idea, ask:

1. Does it improve offline forecasting?
2. Does it improve closed-loop control?
3. Does it remain stable under forecast error?
4. Can it be explained as a control-oriented design choice?

If the answer to only the first question is yes, it is not enough.

