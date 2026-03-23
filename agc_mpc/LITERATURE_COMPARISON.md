# AGC Literature Comparison

## Purpose

This note compares the current `agc_mpc` baseline-first implementation with recent greenhouse forecasting and control papers.

The goal is not to build a fake leaderboard.
The goal is to answer three questions:

1. Are the current AGC results obviously too poor relative to literature?
2. Which literature results are actually comparable to our task?
3. What is still missing in `agc_mpc` compared with stronger forecasting/control systems?

## Our Current Setting

Project:
- `agc_mpc`

Dataset:
- `AutonomousGreenhouseChallenge_edition2`

Task:
- multi-step conditional forecasting for control
- input: `x_past / w_future / u_future`
- output: `Tair / Rhair / CO2air / Tot_PAR`

Current default horizon:
- `24 x 5 min = 2 h`

Current forecasting baselines:
- `GRU`
- `DLinear`
- `SegRNN`
- `Transformer`
- `Transformer-hybrid`

Current control setup:
- surrogate closed-loop benchmark
- `Recorded / GradientMPC / CEMMPC`

Current AGC final-step forecasting results:

| Model | Tair R2 / MAE | Rhair R2 / MAE | CO2air R2 / MAE | Tot_PAR R2 / MAE |
| --- | --- | --- | --- | --- |
| DLinear | 0.9526 / 0.729 | 0.8184 / 4.209 | 0.7928 / 51.481 | 0.9779 / 31.295 |
| Transformer | 0.9413 / 0.823 | 0.7454 / 4.919 | 0.8242 / 47.229 | 0.9859 / 24.964 |
| Transformer-hybrid | 0.9480 / 0.770 | 0.6927 / 5.306 | 0.7434 / 58.318 | 0.9846 / 28.509 |

Interpretation:
- `Tair` is already strong.
- `CO2air` is acceptable to strong for a 2 h multi-step task.
- `Rhair` is the weakest target.
- `Tot_PAR` is very strong offline, but this does not automatically imply best closed-loop behavior.

## Old Strawberry vs AGC

Old project:
- `diffmpc`
- dataset: `Strawberry Greenhouse Environmental Control Dataset(version2).csv`

Direct 2 h comparison figure:
- `results/forecasting/figures/strawberry_vs_agc_dataset_switch.png`

Representative forecast-window figure:
- `results/forecasting/figures/strawberry_vs_agc_forecast_windows.png`

Key comparison on common variables (`Temperature / Humidity / CO2`):

| Setting | Temperature final MAE | Humidity final MAE | CO2 final MAE | Temperature final R2 | Humidity final R2 | CO2 final R2 |
| --- | --- | --- | --- | --- | --- | --- |
| Strawberry old Transformer-hybrid | 3.36 | 6.78 | 105.88 | 0.796 | 0.840 | 0.073 |
| AGC DLinear | 0.76 | 4.46 | 54.73 | 0.949 | 0.798 | 0.776 |
| AGC Transformer | 0.82 | 4.92 | 47.23 | 0.941 | 0.745 | 0.824 |
| AGC Transformer-hybrid | 0.77 | 5.31 | 58.32 | 0.948 | 0.693 | 0.743 |

Takeaway:
- the old Strawberry setup is clearly weaker on the same 2 h horizon framing
- AGC is not “perfect”, but it is much more suitable as a control-oriented benchmark

## Comparable Forecasting Papers

### 1. Ahn et al., 2024

Paper:
- “Evaluating Time-Series Prediction of Temperature, Relative Humidity, and CO2 in the Greenhouse with Transformer-Based and RNN-Based Models”
- Link: https://www.mdpi.com/2073-4395/14/3/417

Task:
- greenhouse climate forecasting
- variables: temperature, RH, CO2
- horizons reported: `1 h` and `3 h`

Models:
- Autoformer
- DLinear
- LSTM
- SegRNN

Reported results:
- 1 h DLinear `R2 = 0.938 / 0.857 / 0.783`
- 3 h DLinear `R2 = 0.833 / 0.680 / 0.580`
- 1 h Autoformer `R2 = 0.744 / 0.636 / 0.590`
- 3 h Autoformer `R2 = 0.554 / 0.411 / 0.488`

Why it matters for us:
- this is the most directly useful forecasting comparison
- it shows that in greenhouse time series, simple models like `DLinear` and `SegRNN` can outperform transformer-style baselines
- our `DLinear` and `Transformer` results on AGC are not obviously below this literature band

What it suggests:
- our current outcome, where `DLinear` is very competitive, is not abnormal
- transformer underperforming a simple baseline in greenhouse forecasting is also not abnormal

### 2. Mao et al., 2024

Paper:
- “A variable weight combination prediction model for climate in a greenhouse based on BiGRU-Attention and LightGBM”
- Link: https://www.sciencedirect.com/science/article/pii/S0168169924002096

Task:
- forecasting `temperature / humidity / PAR`
- horizons from `30 min` to `120 min`

Method:
- `BiGRU-Attention + LightGBM`
- combination weighting optimized by PSO

Reported result at 120 min:
- `R2 = 0.9586 / 0.9232 / 0.8066`

Why it matters for us:
- it is a stronger “engineered prediction stack” than our current baselines
- humidity in that paper is clearly stronger than our current `Rhair`

What is not directly comparable:
- it does not cover `CO2`
- it is a different greenhouse/data regime
- it uses a more specialized hybrid model rather than a clean baseline benchmark

What it suggests:
- if we want stronger humidity performance, a residual/hybrid design is justified
- our current code is still under-optimized relative to stronger forecasting papers

### 3. Guo et al., 2024

Paper:
- “Multi-Step Prediction of Greenhouse Temperature and Humidity Based on Temporal Position Attention LSTM”
- DOI listed in review source: `10.1007/s00477-024-02840-x`
- Search preview used for numbers: https://www.researchgate.net/publication/385934141_Multi-Step_Prediction_of_Greenhouse_Temperature_and_Humidity_Based_on_Temporal_Position_Attention_LSTM

Task:
- temperature and humidity only
- short-to-medium horizon multi-step prediction

Method:
- temporal position attention LSTM
- multiple indoor/outdoor variables

Representative reported result:
- for one 2 h setting, temperature `R2` stays around `0.991 -> 0.704` from near to far steps
- humidity `R2` stays around `0.981 -> 0.816` from near to far steps

Why it matters for us:
- this is an example of a more optimized sequence model than our current baselines
- it mainly tells us that high humidity accuracy is possible with better architecture tuning and task-specific design

What is not directly comparable:
- it does not include `CO2`
- it focuses on temperature/humidity microclimate forecasting, not explicitly control-oriented surrogate modeling

### 4. Cebolla-Alemany et al., 2026

Paper:
- “Thermocast: A modular ensemble learning method for rooftop greenhouse short-term air temperature prediction”
- Link: https://www.sciencedirect.com/science/article/pii/S2772375525009645

Task:
- short-term temperature-only forecasting
- `5 / 10 / 15 min` horizons

Method:
- ensemble with meteorological feature engineering

Reported result:
- `R2 > 0.98`
- `MAE` about `0.280–0.311 °C`

Why it matters for us:
- it shows what excellent short-term single-target performance can look like

Why it is not comparable:
- much shorter horizon
- temperature only
- not a multi-output control surrogate

Takeaway:
- we should not compare our current 2 h multi-output AGC setup against short-term temperature-only papers as if they were equivalent

## Comparable Control Papers

### 5. Mahmood et al., 2021

Paper:
- “Model predictive control strategy for energy efficient greenhouse climate control using machine learning models”
- Link: https://www.sciencedirect.com/science/article/pii/S0959652621033588

Task:
- greenhouse climate control
- focus on temperature tracking and energy efficiency

Method:
- ML model inside MPC
- reported comparison against conventional control

Reported headline result:
- indoor temperature `RMSE` around `0.33–0.36 °C`
- energy reduction around `7.7%` and `16.57%`

Why it matters for us:
- it is a good example of “prediction quality is not the only output; control and energy also matter”

Gap vs our current project:
- we do not yet have economic/resource terms in the control cost
- our current control benchmark is still a surrogate rollout, not a full deployment-grade greenhouse controller

### 6. Chen and You, 2022

Paper:
- “Intelligent control and energy optimization in controlled environment agriculture via nonlinear model predictive control of semi-closed greenhouse”
- Link: https://www.sciencedirect.com/science/article/abs/pii/S0306261922006845

Task:
- simultaneous control of `temperature / humidity / CO2 / light`

Method:
- nonlinear dynamic model from energy and mass balance
- NMPC with multiple actuators and explicit costs

Reported conclusion:
- humidity, CO2, and light can be controlled with almost no violation in case studies
- temperature is generally maintained in acceptable range except extreme summer cases

Why it matters for us:
- this is much closer to the kind of final control paper we actually want to resemble

Gap vs our current project:
- they use a richer mechanistic model
- they optimize true control cost directly
- our current AGC control side is still an early learned-surrogate benchmark

### 7. Kim and You, 2025

Paper:
- “Energy-efficient greenhouse climate control using Gaussian process-based stochastic model predictive control”
- Link: https://www.sciencedirect.com/science/article/pii/S0306261925005719

Task:
- greenhouse climate control under uncertainty

Method:
- GP-SMPC
- online learning for mismatch correction

Reported headline result:
- up to `67%` winter tracking-error reduction
- up to `48%` spring tracking-error reduction
- up to `51.4%` energy and `40%` CO2 cost reduction versus NMPC

Why it matters for us:
- this is a strong example of what uncertainty-aware control papers add beyond baseline MPC

Gap vs our current project:
- no uncertainty model yet
- no online adaptation yet
- no economic evaluation yet

### 8. Mallick et al., 2025

Paper:
- “Reinforcement learning-based model predictive control for greenhouse climate control”
- Link: https://www.sciencedirect.com/science/article/pii/S2772375524003551

Task:
- greenhouse climate control under model uncertainty

Method:
- parametrized MPC learned via RL

Reported conclusion:
- simulation shows improved climate control performance and fewer constraint violations than prior approaches

Why it matters for us:
- it clarifies that stronger modern control papers do not stop at predictor accuracy
- they directly improve the controller or the uncertainty handling

## Bottom-Line Assessment

### What is already defensible

- Switching from old Strawberry to AGC is defensible.
- `DLinear` being very strong is consistent with greenhouse forecasting literature.
- Our current AGC forecasting numbers are not obviously “broken” or dramatically below literature.

### What is clearly still weak

- humidity prediction remains weaker than stronger specialized greenhouse papers
- control side is still early-stage surrogate benchmarking
- no uncertainty-aware forecasting/control
- no economic/resource objective yet

### What is likely limiting current performance in our code

- only `12` training epochs
- no hyperparameter search
- no target-specific loss balancing
- no horizon-aware loss shaping
- no probabilistic forecast head
- no residual hybrid targeted at `Rhair`
- no compartment adapter / transfer-learning layer
- no richer actuator/VIP/physics transition in control rollout

## Recommended Message for Advisor

Use this wording:

- The current AGC results are not yet “final-paper quality”.
- However, they are already within the broad performance band reported by comparable greenhouse forecasting literature.
- The main reason to prefer AGC is not that it already gives perfect scores.
- The main reason is that AGC matches the control-oriented task much better:
  future weather, future control plans, actuator feedback, multiple compartments, and resource signals are all available.
- Therefore AGC is the better research platform for multi-step forecasting plus MPC, even before the modeling stack is fully optimized.
