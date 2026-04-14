# CO2_PAPERS_AND_DIRECTION.md

English canonical version.
Mapped Chinese mirror: [CO2_PAPERS_AND_DIRECTION.zh-CN.md](c:/repositories/strawberry/agc_mpc/CO2_PAPERS_AND_DIRECTION.zh-CN.md)
Last synchronized: `2026-04-07`

## Purpose

This note only focuses on greenhouse `CO2` forecasting and control.

It answers two practical questions:

1. When a paper reports `MAE`, is it computed on normalized data or on physical units?
2. If we want to improve `CO2air` in `agc_mpc`, which papers are worth reading first and which ideas are worth borrowing first?

This is not a ranking for its own sake.
The goal is to extract directions that can really transfer into the current `AGC` workflow.

## A. How To Judge Whether A Reported MAE Is Normalized

In greenhouse papers, there are usually three common cases.

1. The data are normalized for training, but the final error is reported after inverse scaling.
   - These metrics usually carry physical units such as `ppm`, `degC`, or `%RH`.

2. The paper reports error directly on the normalized target.
   - These metrics are usually very small decimals such as `0.0117`, often without physical units.

3. The public abstract page does not provide enough detail.
   - In that case, do not claim the `MAE` is normalized or unnormalized without checking the full paper.

Practical rules:

- If the error is written in `ppm`, it is usually a physical-unit error.
- If the number is very small and unit-free while the target itself is in the hundreds of `ppm`, it is probably normalized error.
- If only `R2` is shown on the public page, the `MAE` status is still unknown.

## B. CO2-Focused Paper List

### B1. Papers That Directly Forecast Greenhouse CO2

| Paper | What problem it solves | Main method | Metric status | What we can borrow | Priority |
| --- | --- | --- | --- | --- | --- |
| [Prediction of CO2 Concentration via Long Short-Term Memory Using Environmental Factors in Greenhouses](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002578287) | Forecast greenhouse `CO2` from environmental factors, `2 h` ahead | `LSTM` | Public abstract mainly reports `R2`, so `MAE` status is unclear from the abstract alone | `CO2` can be modeled as a dedicated target instead of only as one shared head in a general climate model | Medium |
| [Time-serial analysis of deep neural network models for prediction of climatic conditions inside a greenhouse](https://doi.org/10.1016/j.compag.2020.105402) | Jointly forecast `temperature / humidity / CO2` | `ANN`, `NARX`, `RNN-LSTM` | Public results page reports `CO2` error in `ppm`, so this is a physical-unit metric | `CO2` is harder than temperature, and recurrent models still matter in greenhouse dynamics | High |
| [Multi-model fusion method for predicting CO2 concentration in greenhouse tomatoes](https://doi.org/10.1016/j.compag.2024.109623) | Forecast greenhouse tomato `CO2` concentration | `WT + VMD + LSTM + attention + fusion` | Public abstract reports `MAE = 0.0117` and `RMSE = 0.0194` without physical units; most likely normalized error | `CO2` is better handled by decomposition and fusion than by a single backbone | Very High |
| [Prediction of CO2 concentration in mushroom greenhouse via optimized long and short term memory algorithm](https://doi.org/10.1038/s41598-025-86394-0) | Forecast `CO2` in a mushroom greenhouse | `VMD-SSA-LSTM`, `VMD-DBO-LSTM` | Public abstract directly reports `MAE = 2.6365 ppm`, so the metric is in physical units | Even with recurrent backbones, `CO2` benefits strongly from decomposition and optimization | High |
| [Wavelet-decoupled GRU with adaptive attention for multi-step carbon dioxide concentration prediction in intelligent glass greenhouse](https://doi.org/10.1016/j.atech.2025.101653) | Multi-step greenhouse `CO2` forecasting up to `8 h` | wavelet-like decoupling + `GRU` + adaptive attention | Public page reports `ppm` error; training likely uses scaling, but final metrics are physical-unit | Strong support for multi-scale decomposition and adaptive weighting for `CO2` | Very High |

### B2. CO2 Control And Optimization Papers

| Paper | What problem it solves | Main method | Why it matters |
| --- | --- | --- | --- |
| [Model-based control of CO2 concentration in greenhouses at ambient levels increases cucumber yield](https://doi.org/10.1016/j.agrformet.2006.12.002) | `CO2` control near ambient concentration | model-based control using crop uptake modeling | Reminds us the final target is not just accurate `ppm` forecasting, but support for dosing strategy and crop uptake estimation |
| [Model predictive control of a Venlo-type greenhouse system considering electrical energy, water and carbon dioxide consumption](https://doi.org/10.1016/j.apenergy.2021.117163) | Joint energy, water, and `CO2` consumption control | `MPC` | Important if we later move `CO2` from a pure forecast target into the control cost itself |
| [Intelligent control and energy optimization in controlled environment agriculture via nonlinear model predictive control of semi-closed greenhouse](https://doi.org/10.1016/j.apenergy.2022.119334) | Joint control of `temperature / humidity / CO2 / light` | energy- and mass-balance-based `NMPC` | Strong support for modeling the greenhouse as a coupled energy + mass system instead of only as a black-box forecaster |
| [CO2 enrichment in greenhouse production: Towards a sustainable approach](https://doi.org/10.3389/fpls.2022.1029901) | Review of `CO2` enrichment strategy | review | Good entry point if the question shifts from prediction accuracy to sustainable and efficient `CO2` use |

### B3. Gray-Box And Flux-Model Papers

| Paper | What problem it solves | Main method | Why it matters |
| --- | --- | --- | --- |
| [An autocalibrating model for simulating and measuring net canopy photosynthesis using a standard greenhouse climate computer](https://doi.org/10.1016/0168-1699(91)90019-6) | Estimate net canopy photosynthesis inside a greenhouse | `CO2` balance model + black-box photosynthesis model | This is one of the clearest precedents for a `CO2 balance + black-box` gray-box route |
| [Estimation of net photosynthesis of a greenhouse canopy using a mass balance method and mechanistic models](https://doi.org/10.1016/0168-1923(94)90106-6) | Estimate canopy photosynthesis from greenhouse `CO2` balance | mass balance + mechanistic models | Supports modeling `CO2` together with canopy uptake and ventilation exchange instead of as an ordinary scalar time series |
| [Validation of a Photosynthesis Model through the Use of the CO2 Balance of a Greenhouse Tomato Canopy](https://doi.org/10.1006/anbo.1999.0938) | Validate a photosynthesis model using greenhouse `CO2` balance | `CO2` balance + plant-physiology model | Reinforces the idea that `CO2` should be linked to plant uptake processes |

## C. What These Papers Jointly Suggest

Across direct `CO2` forecasting papers, two patterns are stable.

1. `CO2` is more non-stationary than `Tair` and more regime-dependent.
2. Methods that work well for `CO2` usually add at least one of the following:
   - decomposition / denoising / multi-scale processing
   - dynamic fusion / adaptive weighting

This is consistent with what we already see in `AGC`:

- `CO2air` may look acceptable on global average metrics
- but local rollout windows can still drift badly

So the next step should not be just "swap in a bigger generic transformer." A more realistic choice is one of the two routes below.

## D. Two Practical Routes For `agc_mpc`

### Route 1: A CO2-Specialized Forecasting Branch

Keep the current multi-target setup, but add a more specialized branch for `CO2` than the current residual variants.

The most reasonable ingredients from the literature are:

1. decomposition before sequence modeling
   - `WT`
   - `VMD`
   - or other multi-scale splits

2. a more suitable backbone for the `CO2` branch
   - `GRU`
   - `LSTM`
   - `GRU/LSTM + attention`

3. variable-weight fusion
   - weighting by target variable
   - weighting by horizon
   - weighting by context

This route is the easiest one to integrate into the current forecasting codebase.

### Route 2: An Energy-Water-Carbon Gray-Box Model

Instead of treating `CO2air` as just another output channel, define the greenhouse as a coupled system of:

- energy flow
- water flow
- carbon flow

Then build a gray-box predictor:

- use mechanistic balance equations where they are known
- use a black-box residual model where the physics is incomplete

For `CO2`, natural latent quantities include:

- `CO2 dosing`
- ventilation exchange
- canopy net uptake / photosynthesis
- respiration terms

This route is more research-oriented and more greenhouse-native.

## E. Suggested Reading Order

If the current goal is to improve `CO2air` forecasting quickly:

1. [Multi-model fusion method for predicting CO2 concentration in greenhouse tomatoes](https://doi.org/10.1016/j.compag.2024.109623)
2. [Wavelet-decoupled GRU with adaptive attention for multi-step carbon dioxide concentration prediction in intelligent glass greenhouse](https://doi.org/10.1016/j.atech.2025.101653)
3. [Prediction of CO2 concentration in mushroom greenhouse via optimized long and short term memory algorithm](https://doi.org/10.1038/s41598-025-86394-0)
4. [Time-serial analysis of deep neural network models for prediction of climatic conditions inside a greenhouse](https://doi.org/10.1016/j.compag.2020.105402)

If the goal is to move toward a stronger greenhouse-native CO2 modeling line:

1. [An autocalibrating model for simulating and measuring net canopy photosynthesis using a standard greenhouse climate computer](https://doi.org/10.1016/0168-1699(91)90019-6)
2. [Model-based control of CO2 concentration in greenhouses at ambient levels increases cucumber yield](https://doi.org/10.1016/j.agrformet.2006.12.002)
3. [Intelligent control and energy optimization in controlled environment agriculture via nonlinear model predictive control of semi-closed greenhouse](https://doi.org/10.1016/j.apenergy.2022.119334)

## Summary

For `CO2air`, the literature does not support the claim that another generic backbone swap will solve the problem.

The stronger directions are:

1. `CO2`-specialized `decomposition + sequence model + dynamic fusion`
2. `CO2 balance + photosynthesis + control` gray-box modeling

If we keep the current `agc_mpc` architecture, the fastest next step is Route 1.
If the goal is a more original and greenhouse-native research line, Route 2 is stronger.