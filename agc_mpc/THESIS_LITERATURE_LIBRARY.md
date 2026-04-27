# Thesis Literature Library

English canonical version.
Mapped Chinese mirror: [THESIS_LITERATURE_LIBRARY.zh-CN.md](c:/repositories/strawberry/agc_mpc/THESIS_LITERATURE_LIBRARY.zh-CN.md)
Last synchronized: `2026-04-20`

## Purpose

This document is the paper-facing literature library for the `agc_mpc` thesis line.

It is broader than the previous control-relevant MPC note. It should collect any paper that may later be cited in the thesis, including:

- greenhouse multi-step forecasting
- greenhouse climate control and economic MPC
- CO2 forecasting, carbon balance, and CO2 enrichment
- general time-series forecasting architectures
- control-relevant identification, validation, and prediction-control correlation
- uncertainty, robustness, probabilistic forecasting, and resource-aware control

This is not a leaderboard. Direct metric comparison is only valid when the task, horizon, target variables, dataset regime, and control objective are sufficiently close.

## Reading Guide

| Label | Meaning | How to use |
| --- | --- | --- |
| Most comparable | Greenhouse multi-variable forecasting or greenhouse climate control with similar targets/horizons. | Use in main literature comparison and motivation. |
| Partially comparable | Greenhouse papers with shorter horizons, fewer targets, or different objectives. | Use as directional evidence, not as direct numerical benchmarks. |
| Reference only | General time-series, control theory, or building/HVAC papers. | Use to justify architecture, validation metrics, or control methodology. |

## Current Thesis Story Supported By This Library

1. Greenhouse forecasting is not a plain Transformer leaderboard problem. Recent greenhouse and general time-series papers support hybrid, residual, decomposition, normalization, attention/RNN, and horizon-aware fusion designs.
2. CO2 is a harder and more regime-dependent target than `Tair`; it benefits from decomposition, multi-scale modeling, adaptive fusion, and eventually carbon-balance gray-box modeling.
3. Better offline forecasting does not universally imply better closed-loop MPC. The relation is often positive inside a fixed setup, but cross-model selection requires control-relevant validation.
4. Our current PHF line should be written as a forecasting improvement plus a control-relevant validation story, not as a claim that the offline forecast leader is automatically the control leader.
5. A mature greenhouse MPC paper eventually needs uncertainty handling, constraints, resource/economic costs, and closed-loop validation, not only point forecast accuracy.

## Key Terms

| Term | Meaning | Why it matters here |
| --- | --- | --- |
| Controlled environment agriculture, CEA | Crop production in controlled spaces such as greenhouses or plant factories. | The final thesis domain is greenhouse climate forecasting and control. |
| Multi-step forecasting | Predicting several future time steps at once or recursively. | Our default AGC horizon is `24 x 5 min = 2 h`. |
| Exogenous variables | Future drivers not controlled by the model, such as weather. | `Weather.csv` gives future weather that should enter prediction and MPC. |
| Control inputs | Variables selected by the controller, such as setpoints. | `u_future` represents planned greenhouse control actions. |
| Residual model | A model that learns corrections on top of a stable baseline. | Our strongest line uses stable main paths plus residual/specialist correction. |
| Decomposition | Splitting a series into trend, seasonal, frequency, or multi-scale parts. | CO2 and greenhouse signals often contain slow cycles plus short disturbances. |
| Horizon-aware fusion | Varying model trust by prediction step. | PHF uses horizon-dependent trust and terminal pullback. |
| Model predictive control, MPC | A controller that predicts a finite horizon, optimizes future actions, applies the first action, and repeats. | This is the active control mainline. |
| Receding horizon | MPC applies only the first optimized move before re-solving. | First-step and short-horizon errors can dominate closed-loop behavior. |
| Control-relevant validation | Validating a model by its intended control use, not only generic prediction fit. | This motivates `control_relevant_validation.py`. |
| Oracle gap | Gap between realistic MPC and an ideal controller with perfect future information. | Useful to quantify remaining forecast-related control loss. |

## Project Positioning Against Literature

This section consolidates the useful thesis-positioning material from the older [LITERATURE_COMPARISON.md](c:/repositories/strawberry/agc_mpc/LITERATURE_COMPARISON.md). That older file should be treated as a snapshot; this library is the maintained paper-facing version.

### Current AGC setting

| Item | Current setting |
| --- | --- |
| Project | `agc_mpc` |
| Main dataset | `AutonomousGreenhouseChallenge_edition2` |
| Task | control-oriented multi-step conditional forecasting |
| Inputs | `x_past / w_future / u_future` |
| Outputs | `Tair / Rhair / CO2air / Tot_PAR`, with recent fair-budget work often using `Tair / Rhair / CO2air` |
| Default horizon | `24 x 5 min = 2 h` |
| Forecasting baselines | `GRU`, `DLinear`, `SegRNN`, `Transformer`, `Transformer-hybrid`, residual and CO2-specialist variants |
| Control setup | surrogate closed-loop benchmark with `Recorded / GradientMPC / CEMMPC` |

### Early AGC baseline position

Older AGC baseline results showed the following final-step pattern:

| Model | Tair R2 / MAE | Rhair R2 / MAE | CO2air R2 / MAE | Tot_PAR R2 / MAE |
| --- | --- | --- | --- | --- |
| `DLinear` | `0.9526 / 0.729` | `0.8184 / 4.209` | `0.7928 / 51.481` | `0.9779 / 31.295` |
| `Transformer` | `0.9413 / 0.823` | `0.7454 / 4.919` | `0.8242 / 47.229` | `0.9859 / 24.964` |
| `Transformer-hybrid` | `0.9480 / 0.770` | `0.6927 / 5.306` | `0.7434 / 58.318` | `0.9846 / 28.509` |

Interpretation:

- `Tair` was already strong.
- `CO2air` was acceptable to strong for a 2 h multi-step task.
- `Rhair` was the weakest target.
- `Tot_PAR` was very strong offline, but strong offline light prediction does not automatically imply best closed-loop control.
- The later CO2-specialist work improved the CO2 frontier, so these older numbers should be used only for historical positioning, not as the current final benchmark.

### Old Strawberry vs AGC

The older strawberry dataset remains useful as a stress-test reference, but AGC is a stronger thesis platform because it contains future weather, future control plans, actuator feedback, multiple compartments, and resource signals.

Representative 2 h comparison on common variables:

| Setting | Temperature final MAE | Humidity final MAE | CO2 final MAE | Temperature final R2 | Humidity final R2 | CO2 final R2 |
| --- | --- | --- | --- | --- | --- | --- |
| old Strawberry `Transformer-hybrid` | `3.36` | `6.78` | `105.88` | `0.796` | `0.840` | `0.073` |
| AGC `DLinear` | `0.76` | `4.46` | `54.73` | `0.949` | `0.798` | `0.776` |
| AGC `Transformer` | `0.82` | `4.92` | `47.23` | `0.941` | `0.745` | `0.824` |
| AGC `Transformer-hybrid` | `0.77` | `5.31` | `58.32` | `0.948` | `0.693` | `0.743` |

Thesis interpretation:

- The old Strawberry setup is clearly weaker under the same 2 h framing.
- AGC is not perfect, but it is much more suitable for a control-oriented benchmark.
- The main reason to prefer AGC is not that its scores are perfect; it is that the data interface matches MPC much better.

### Bottom-line literature assessment

Defensible claims:

- Switching from the old Strawberry dataset to AGC is justified.
- `DLinear` being strong is consistent with greenhouse and long-time-series forecasting literature.
- Current AGC forecasting numbers are not obviously broken or far outside the broad band of comparable greenhouse papers.

Remaining weaknesses:

- Humidity prediction remains weaker than stronger specialized greenhouse forecasting papers.
- Closed-loop control is still an early-stage surrogate benchmark.
- The project does not yet have uncertainty-aware forecasting/control.
- The project does not yet have a mature economic/resource objective.

Likely limiting factors:

- limited hyperparameter search relative to specialized forecasting papers
- limited target-specific loss balancing
- limited horizon-aware loss shaping
- no probabilistic forecast head yet
- no explicit humidity-focused specialist branch
- no compartment adapter or transfer-learning layer
- no richer actuator/VIP/physics transition in the current control rollout

Advisor-facing summary:

> The current AGC results are not yet final-paper quality, but they are already within the broad performance band reported by comparable greenhouse forecasting literature. The main reason to prefer AGC is that it matches the control-oriented task: future weather, future control plans, actuator feedback, multiple compartments, and resource signals are all available. Therefore, AGC is the better research platform for multi-step forecasting plus MPC, even before the modeling stack is fully optimized.

## A. Greenhouse Forecasting Papers

| Paper | Task | Method | Baselines | Main takeaway | Link | Comparability |
| --- | --- | --- | --- | --- | --- | --- |
| Ahn et al., 2024 | Greenhouse `temperature / RH / CO2`, `1 h` and `3 h` forecasts | `Autoformer` | `DLinear`, `LSTM`, `SegRNN` | One of the closest greenhouse forecasting references; simple models such as `DLinear` and `SegRNN` can be strong, and Transformer is not automatically superior. | https://www.mdpi.com/2073-4395/14/3/417 | Most comparable |
| Li et al., 2024 | Greenhouse air and soil temperature, `30-480 min` | `Attention-LSTM` | `RNN`, `GRU`, `LSTM` | RNN plus attention remains a valid greenhouse route, especially for short/medium horizons. | https://www.sciencedirect.com/science/article/pii/S0168169923009900 | Partially comparable |
| Mao et al., 2024 | Greenhouse `temperature / humidity / PAR`, `30-120 min` | `PSO-BiGRU-Attention-LightGBM` | `BiGRU-Attention`, `LightGBM`, equal-weight ensemble | Strong evidence for hybrid and variable-weight fusion rather than a single backbone. | https://www.sciencedirect.com/science/article/pii/S0168169924002096 | Partially comparable |
| Guo et al., 2024 | Greenhouse temperature and humidity multi-step prediction | temporal-position-attention `LSTM` | LSTM-family comparisons | Shows that high humidity accuracy is possible with task-specific sequence modeling, but it does not include CO2 or closed-loop control. | https://doi.org/10.1007/s00477-024-02840-x | Partially comparable |
| Wang et al., 2025 | Greenhouse soil temperature, `3 h / 6 h / 24 h / 48 h` | `ReSSA-iTransformer` = `iTransformer + RevIN + SSA` | `LSTM`, `Informer`, `Autoformer`, `iTransformer` | Strong Transformer variants usually include normalization, decomposition, or task-specific modification. | https://www.mdpi.com/2073-4395/15/1/223 | Reference/partial |
| Choi and Yang, 2025 | Greenhouse `temperature / RH / CO2` probabilistic forecasting, `3 h` | Probabilistic `1D CNN`, probabilistic `LSTM` | deterministic CNN/LSTM | Forecast uncertainty is important if forecasting is meant to support control. | https://www.mdpi.com/2077-0472/15/23/2461 | Most comparable for uncertainty |
| Seri et al., 2025 | Greenhouse microclimate modeling with variable coupling | Directed `STGNN` | `RNN` | Structured coupling can matter more than a pure temporal backbone when actuator-climate links are explicit. | https://www.sciencedirect.com/science/article/pii/S0360132325009461 | Reference/partial |
| Cebolla-Alemany et al., 2026 | Rooftop greenhouse air temperature, `5 / 10 / 15 min` | `Thermocast` modular ensemble | traditional and ensemble regressors | Excellent short-horizon temperature-only results are useful as an upper reference, but not a direct benchmark for 2 h multi-output AGC forecasting. | https://www.sciencedirect.com/science/article/pii/S2772375525009645 | Partially comparable |

### Detailed Note: Ahn et al., 2024

Paraphrased summary:

This paper evaluates time-series predictors for greenhouse temperature, relative humidity, and CO2. It is directly relevant because it covers the same core environmental variables as our AGC forecasting task. Its important message is that greenhouse time-series forecasting does not automatically favor Transformer-style models; linear and recurrent baselines can remain competitive.

Thesis use:

- Cite when explaining why `DLinear` being strong in AGC is not surprising.
- Cite when arguing for stable baselines and residual/specialist corrections instead of plain backbone swapping.

### Detailed Note: Mao et al., 2024

Paper:
- Xiaojuan Mao et al., 2024
- `A variable weight combination prediction model for climate in a greenhouse based on BiGRU-Attention and LightGBM`
- Link: https://www.sciencedirect.com/science/article/pii/S0168169924002096

What the paper does:

| Item | Content |
| --- | --- |
| Greenhouse | Single Venlo-type greenhouse in Nanjing, China |
| Crop | Cherry tomato |
| Time range | `2020-09-23` to `2021-06-06` |
| Sampling interval | `10 min` |
| Sample count | `37,008` |
| Inputs | indoor climate, outdoor climate, and control-operation variables |
| Targets | air temperature, air humidity, PAR |
| History length | `120 min` |
| Horizons | `30-120 min` |
| Single models | `GRU`, `BiGRU`, `BiGRU-Attention`, `XGBoost`, `LightGBM` |
| Ensemble models | equal-weight `BiGRU-Attention-LightGBM`, variable-weight `PSO-BiGRU-Attention-LightGBM` |
| Metrics | `RMSE`, `MAE`, `R2` |

Why its high R2 is not directly comparable to AGC:

- It does not include CO2.
- It uses a single greenhouse and shorter horizons.
- It is optimized around forecasting only, not MPC surrogate rollout.
- Its hybrid model already includes sequence modeling, tabular learning, and horizon-dependent weighting.

Thesis use:

- Cite to justify hybrid/residual/fusion designs.
- Cite to support horizon-dependent weighting.
- Do not use its metric values as a direct benchmark against AGC `CO2air`.

### Detailed Note: Guo et al., 2024

Paper:
- `Multi-Step Prediction of Greenhouse Temperature and Humidity Based on Temporal Position Attention LSTM`
- DOI: https://doi.org/10.1007/s00477-024-02840-x

Paraphrased summary:

This paper focuses on temperature and humidity multi-step prediction using a temporal-position-attention LSTM design with indoor and outdoor variables. It is useful as evidence that stronger humidity forecasting is possible when the architecture is tuned to the greenhouse task, but it is not a direct benchmark for our current AGC setup because it does not include CO2 and is not evaluated as a control surrogate.

Thesis use:

- Cite when discussing the current weakness of `Rhair` relative to specialized greenhouse forecasting papers.
- Use as motivation for future humidity-focused residual or specialist branches.
- Do not use it as direct evidence for MPC performance.

### Detailed Note: Choi and Yang, 2025

Paraphrased summary:

This paper studies probabilistic forecasting for greenhouse climate variables including CO2. Its relevance is not only its model family, but its uncertainty framing. For control, a predictor should ideally report uncertainty or risk, because MPC decisions can be sensitive to future disturbances and constraint violations.

Thesis use:

- Cite in future work for probabilistic PHF or stochastic MPC.
- Cite when explaining why point forecast MAE is not the final endpoint.

## B. Greenhouse Control And Economic MPC Papers

| Paper | Task | Method | Baseline / comparison | Main takeaway | Link | Use |
| --- | --- | --- | --- | --- | --- | --- |
| Svensen et al., 2024 | Greenhouse production control under parameter uncertainty | chance-constrained `SMPC` | nonlinear chance-constrained MPC setup | Control papers compete on uncertainty, constraints, and tractability, not just predictor names. | https://www.sciencedirect.com/science/article/pii/S0168169923009663 | uncertainty-aware control |
| Le and Bui, 2025 | Smart greenhouse `NMPC`, `30` day simulation | `NMPC` + `LSTM` disturbance forecast | feedback-only, forecast preview, ideal preview, cold/warm start | Very useful for comparing forecast preview, ideal preview, and warm start. | https://www.mdpi.com/2076-3417/15/14/7988 | control validation design |
| Mallick et al., 2025 | Greenhouse climate control under prediction uncertainty | RL-based MPC | robust/stochastic MPC positioning | Represents controller-learning direction; useful after the surrogate benchmark is stable. | https://www.sciencedirect.com/science/article/pii/S2772375524003551 | future work |
| Kim and You, 2025 | Energy-efficient greenhouse climate control under uncertainty | `GP-SMPC` + online learning | `NMPC`, `RMPC`, `DDRMPC` framing | Mature control work combines uncertainty, online correction, energy, and CO2 costs. | https://www.sciencedirect.com/science/article/pii/S0306261925005719 | target final control story |
| Mansour et al., 2025 | Semi-closed greenhouse climate control, robustness, economic optimization, transfer | hierarchical MPC + DRL | robust/stochastic MPC, model-free DRL | A later-stage system shape: upper economic layer plus lower tracking control. | https://www.sciencedirect.com/science/article/pii/S2772375525005581 | future architecture |
| Mahmood et al., 2021 | Greenhouse temperature control and energy saving | ML model + MPC | conventional control | Control output must include energy/resource outcome, not tracking only. | https://www.sciencedirect.com/science/article/pii/S0959652621033588 | economic motivation |
| Chen and You, 2022 | Semi-closed greenhouse `temperature / humidity / CO2 / light` control | energy/mass-balance `NMPC` | case comparisons | Mature greenhouse NMPC is multivariable, multi-actuator, and explicitly economic. | https://www.sciencedirect.com/science/article/abs/pii/S0306261922006845 | greenhouse-native MPC |

## C. CO2 Forecasting, Carbon Balance, And CO2 Enrichment Papers

| Paper | Task | Method | Metric note | Main takeaway | Link | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM with environmental factors for greenhouse CO2 | Forecast greenhouse CO2 `2 h` ahead | `LSTM` | public abstract mainly reports `R2`; MAE status unclear | CO2 should be treated as a dedicated target, not only a shared head. | https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002578287 | Medium |
| Time-serial analysis of DNN models for greenhouse climate | Joint `temperature / humidity / CO2` forecasting | `ANN`, `NARX`, `RNN-LSTM` | public page reports CO2 error in `ppm` | CO2 is harder than temperature, and recurrent models remain relevant. | https://doi.org/10.1016/j.compag.2020.105402 | High |
| Multi-model fusion for greenhouse tomato CO2 | Greenhouse tomato CO2 concentration | `WT + VMD + LSTM + attention + fusion` | public abstract reports small unit-free MAE/RMSE, likely normalized | CO2 benefits from decomposition and adaptive fusion. | https://doi.org/10.1016/j.compag.2024.109623 | Very high |
| Mushroom greenhouse CO2 optimized LSTM | Mushroom greenhouse CO2 | `VMD-SSA-LSTM`, `VMD-DBO-LSTM` | reports ppm-scale MAE | Decomposition plus optimization can strongly improve CO2 forecasting. | https://doi.org/10.1038/s41598-025-86394-0 | High |
| Wavelet-decoupled GRU with adaptive attention | Multi-step greenhouse CO2 up to `8 h` | wavelet-like decoupling + `GRU` + adaptive attention | public page reports physical ppm errors | Strong support for multi-scale decomposition and adaptive weighting for CO2. | https://doi.org/10.1016/j.atech.2025.101653 | Very high |
| Model-based CO2 control increases cucumber yield | Greenhouse CO2 control near ambient concentration | model-based control with crop uptake | control/yield paper | CO2 control should ultimately relate to dosing strategy and crop uptake. | https://doi.org/10.1016/j.agrformet.2006.12.002 | High |
| MPC of Venlo greenhouse considering energy, water, CO2 | Joint resource control | MPC | control/economic paper | CO2 belongs in the resource/economic objective, not only as a forecast target. | https://doi.org/10.1016/j.apenergy.2021.117163 | High |
| CO2 enrichment review | Review of greenhouse CO2 enrichment | review | review | Useful background for sustainable CO2 use and biological interpretation. | https://doi.org/10.3389/fpls.2022.1029901 | Medium |
| Autocalibrating canopy photosynthesis model | Estimate net canopy photosynthesis | CO2 balance + black-box photosynthesis | gray-box | Strong precedent for `CO2 balance + black-box residual`. | https://doi.org/10.1016/0168-1699(91)90019-6 | High |
| Net photosynthesis by mass balance | Estimate canopy photosynthesis | mass balance + mechanistic model | gray-box | Supports linking CO2 to canopy uptake and ventilation exchange. | https://doi.org/10.1016/0168-1923(94)90106-6 | High |
| Photosynthesis model validation using CO2 balance | Validate photosynthesis model | CO2 balance + plant physiology | gray-box | Reinforces greenhouse-native CO2 modeling. | https://doi.org/10.1006/anbo.1999.0938 | High |

## D. General Time-Series Architecture References

These papers are not greenhouse-specific. They are useful for explaining why certain model families are imported into `agc_mpc`.

| Paper | Task | Method | Main takeaway | Link | Use |
| --- | --- | --- | --- | --- | --- |
| Zeng et al., 2022 | Long-term time-series forecasting | `LTSF-Linear / DLinear` | Simple linear models can beat many Transformer variants; supports stable linear main paths. | https://arxiv.org/abs/2205.13504 | DLinear baseline justification |
| PatchTST, 2023 | Long-term time-series forecasting | patch-based Transformer | Patching changes tokenization and often improves TS Transformers. | https://openreview.net/forum?id=Jbdc0vTOcol | future residual candidate |
| iTransformer, 2024 | Multivariate time-series forecasting | inverted tokenization by variable | Useful for variable coupling such as `Tair / Rhair / CO2air / PAR`. | https://openreview.net/forum?id=JePfAI8fah | current residual backbone |
| TimeMixer, 2024 | Long-term forecasting | multi-scale mixing | Supports multi-scale decomposition/fusion for greenhouse cycles and disturbances. | https://openreview.net/forum?id=7oLshfEIC2 | future multi-scale route |
| SAMformer, 2024 | Multivariate forecasting | shallow Transformer + SAM | Transformer weakness can be training-related, not only architectural. | https://openreview.net/forum?id=8kLzL5QBh2 | training-strategy citation |
| ETSformer, 2023 | Long-term forecasting | level-growth-seasonality decomposition | Supports decomposition-style Transformer designs. | https://openreview.net/forum?id=5m_3whfo483 | decomposition support |
| FreTS, 2023 | Time-series forecasting | frequency-domain MLP | Frequency-domain modeling can capture global periodic structure compactly. | https://openreview.net/forum?id=iif9mGCTfy | future frequency residual |
| TiDE, 2023 | Long-term forecasting | dense encoder-decoder | Strong alternatives to Transformers can be MLP-style. | https://arxiv.org/abs/2304.08424 | architecture baseline |
| Mamba, 2024 | sequence modeling | selective state space model | Dynamic systems with long memory may benefit from SSM-style modeling. | https://arxiv.org/abs/2312.00752 | future dynamic backbone |
| Simple-Mamba, 2024 | time-series forecasting | Mamba variant | Mamba potential is task- and implementation-dependent; do not blindly replace DLinear. | https://arxiv.org/abs/2403.11144 | cautionary citation |
| OneNet, 2023 | online time-series forecasting | online ensemble | Supports dynamic combination of models with different inductive biases. | https://openreview.net/forum?id=Q25wMXsaeZ | expert fusion support |

## E. Control-Relevant MPC And Prediction-Control Correlation

### E1. Main conclusion

If a model predicts better offline, it does not necessarily control better in closed loop.

There is often a positive relation within a fixed task, model class, objective, horizon, and tuning setup. However, there is no universal monotonic rule saying that lower generic forecast MAE/MSE or higher R2 always implies better MPC performance. MPC performance depends on where the error occurs in the horizon, whether the model preserves control-sensitive input-output directions, whether the optimizer sees useful gradients or sensitivities, and whether forecast errors affect active constraints and economic terms.

### E2. Control-relevant MPC literature map

| Paper | Domain | Main message | Use in our paper |
| --- | --- | --- | --- |
| [Huang, Malhotra, and Tamayo, 2003](https://doi.org/10.1016/S0009-2509(03)00077-0) | process MPC identification and validation | A model should be validated for multi-step predictive control use, not only generic model fit. | Foundational support for control-relevant validation. |
| [Misra, Darby, Panjwani, and Nikolaou, 2017](https://doi.org/10.3390/pr5030042) | multivariable control-relevant experiment design | Accurate models must also satisfy control-relevant properties such as integral controllability. | Supports the claim that model closeness alone is not enough. |
| [Badwe et al., 2010](https://doi.org/10.1016/j.jprocont.2009.12.006) | MPC model-plant mismatch diagnostics | A poor model does not always degrade control, and mismatch impact depends on setpoint directions. | Supports non-universal, direction-dependent prediction-control relation. |
| [Lawrynczuk, 2010](https://doi.org/10.1016/j.neucom.2009.12.015) | neural models for predictive control | Neural predictors should be trained with their MPC role in mind, especially long-range prediction. | Supports control-aware training/validation beyond one-step loss. |
| [Potts, Romano, and Garcia, 2014](https://doi.org/10.1016/j.conengprac.2013.09.007) | MPC relevant identification | Multi-step prediction error methods are useful under model-structure mismatch. | Supports horizon-aware model selection when the true system is outside the model class. |
| [Ludolfinger, Hamacher, and Martens, 2025](https://doi.org/10.1016/j.segy.2025.100202) | smart energy storage MPC | Forecast MSE and MPC revenue can disagree; perfect-forecast oracle remains much better. | Strong modern empirical support that prediction metrics alone can misrank MPC models. |
| [Wang, Mai, Li, and Ding, 2024](https://doi.org/10.3390/buildings14072212) | HVAC demand response MPC | MPC performance generally worsens as prediction accuracy drops, but even lower-accuracy models can still be useful. | Supports a conditional positive relation within a controlled setting. |
| [Hou, Li, Nord, and Huang, 2022](https://doi.org/10.1016/j.enbuild.2021.111793) | building HVAC MPC under weather uncertainty | Weather forecast correction can recover most theoretical MPC benefit. | Supports bias/error correction as a control-relevant forecast operation. |
| [Jain et al., 2018](https://doi.org/10.1186/s42162-018-0064-9) | occupancy prediction errors in HVAC MPC | Larger occupancy prediction errors can make MPC worse than a simple static schedule; robustness layers can mitigate this. | Supports robustness evaluation under prediction errors. |
| [Esrafilian-Najafabadi and Haghighat, 2022](https://doi.org/10.1016/j.enbuild.2021.111808) | occupancy model selection for HVAC control | MAE/accuracy have only weak to moderate correlation with control-oriented performance scores. | Supports replacing pure ML metrics with control scores. |
| [Grant and Gehbauer, 2022](https://doi.org/10.26868/25746308.2022.C026) | dynamic facades MPC under weather forecast error | Forecast inaccuracy can increase cost and glare, while bias correction can largely remove the penalty. | Supports signed bias and corrected forecast diagnostics. |

### E3. Detailed control-relevant notes

#### Huang, Malhotra, and Tamayo, 2003

Paraphrased abstract:

The paper studies how process data should be filtered and validated when the model will be used inside MPC. Instead of asking whether a model is generically accurate, it asks whether the model supports optimal predictions across the MPC prediction horizon. It develops multi-step prediction filters and a validation scheme that detects changes relevant to predictive control.

Key conclusions:

- A model can fail strict generic validation but still control acceptably.
- A model can pass generic validation but fail to deliver good control.
- Validation should target the intended use of the model.
- For MPC, that intended use is multi-step prediction inside a finite-horizon optimizer.

Use in thesis:

> Following the control-relevant validation view of Huang et al., model assessment for MPC should be tied to multi-step prediction quality over the controller's finite horizon rather than to generic offline fit alone.

#### Misra, Darby, Panjwani, and Nikolaou, 2017

Paraphrased abstract:

This review explains that models used for multivariable control must satisfy both ordinary accuracy requirements and control-specific requirements. It focuses on experiment design for estimating model order and identifying models that satisfy integral controllability, a property related to robust multivariable control.

Key conclusions:

- Standard experiment design may not generate data that are best for controller design.
- A model that is close to the real process in an ordinary error sense may still be unsuitable for robust control.
- Multivariable systems need control-relevant properties, not only small output error.

Use in thesis:

> In multivariable control, model quality is not reducible to input-output fit; the identified model must also preserve the control-relevant properties needed by the controller.

#### Badwe et al., 2010

Paraphrased abstract:

The paper studies how model-plant mismatch affects MPC performance using closed-loop data. It emphasizes that poor control may be caused by mismatch, disturbances, or tuning, and that the impact of mismatch is not uniform. The effect depends on the directions in which setpoints move.

Key conclusions:

- A poor model does not necessarily degrade closed-loop control.
- Poor control is not necessarily caused by model error alone.
- The impact of model mismatch can depend on setpoint directions.
- Closed-loop diagnostics are required to isolate when mismatch is actually responsible for performance loss.

Use in thesis:

> Model mismatch matters through its closed-loop effect and can be direction dependent; therefore prediction error must be interpreted together with control actions and setpoint trajectories.

#### Lawrynczuk, 2010

Paraphrased abstract:

The paper connects neural model training with the later use of that model inside MPC. It argues that the model identification stage should account for the fact that MPC repeatedly uses the model for multi-step prediction and optimization, not merely one-step prediction.

Key conclusions:

- Neural predictors used in MPC should not be trained as generic one-step predictors only.
- Recursive multi-step errors can accumulate and mislead MPC.
- Training and validation should reflect the model's future predictive-control role.

Use in thesis:

> Neural surrogate models for MPC should be trained and evaluated according to the multi-step predictions used by the controller, rather than only by conventional one-step prediction error.

#### Potts, Romano, and Garcia, 2014

Paraphrased abstract:

The paper studies MPC-relevant identification when there is model-structure mismatch, meaning the chosen model class cannot exactly represent the real process. It proposes an enhanced multi-step prediction error method and emphasizes predictor stability and robustness.

Key conclusions:

- MPC-relevant identification is especially useful when the model structure is imperfect.
- Optimizing disturbance and prediction behavior over the horizon can improve control-relevant fit.
- Stability of the predictor matters, not only error magnitude.

Use in thesis:

> When the true process is outside the selected model class, MPC-relevant multi-step identification and validation become more appropriate than ordinary prediction-error minimization alone.

#### Ludolfinger, Hamacher, and Martens, 2025

Paraphrased abstract:

The paper compares forecasting models for load, photovoltaic generation, and electricity price in a smart energy storage MPC setup. It evaluates both test-set prediction errors and MPC revenue. It finds that the model with the best MSE can perform best in control, but other models show a mismatch between prediction ranking and control ranking.

Key conclusions:

- XGBoost achieved the lowest reported MSE and the highest realistic revenue gain.
- A linear model with poor MSE ranking still achieved competitive MPC revenue.
- Perfect forecasts achieved a much higher oracle gain than realistic forecasts.
- Standard MSE may misrepresent the value of forecasts for control.

Use in thesis:

> Recent energy-storage MPC evidence shows that standard forecast error metrics can misrepresent control utility, and that realistic controllers may remain far from a perfect-forecast oracle even when test errors appear strong.

#### Wang, Mai, Li, and Ding, 2024

Paraphrased abstract:

The paper compares SVM, ANN, XGBoost, and LightGBM predictors for HVAC demand response MPC. It evaluates prediction accuracy, prediction time, and training time, then changes model accuracy levels to test how MPC performance responds.

Key conclusions:

- Within the paper's controlled degradation experiment, lower prediction accuracy generally reduced MPC performance.
- It was still worth using MPC even at relatively lower prediction accuracy.
- Prediction speed and training speed matter for MPC deployment.

Use in thesis:

> Prediction accuracy can improve MPC performance within a fixed control setup, but this does not remove the need for control-relevant validation across different model structures.

#### Hou, Li, Nord, and Huang, 2022

Paraphrased abstract:

The paper studies building HVAC MPC when weather forecasts are uncertain. It introduces a simple forecast error model to improve weather inputs before MPC optimization and compares against conventional rule-based control and MPC without forecast-error correction.

Key conclusions:

- Weather forecast errors can severely reduce MPC benefit.
- Adding an error model recovered most of the theoretical energy-cost and comfort benefit in their case.
- Bias/error correction can be a control-relevant operation, not merely a forecasting improvement.

Use in thesis:

> Forecast correction should be evaluated by its effect on the controller's objective and constraints, because reducing the right error can recover MPC benefit.

#### Jain et al., 2018

Paraphrased abstract:

The paper studies how occupancy prediction errors affect HVAC MPC. Using a building thermal simulator and real occupancy data, it shows that larger occupancy prediction errors can degrade both energy use and comfort. It also evaluates a personal environmental comfort layer that makes the control system more robust to prediction errors.

Key conclusions:

- MPC can become worse than a simple static schedule when occupancy prediction errors grow.
- Prediction errors affect both energy and comfort.
- Robustness layers can mitigate the effect of forecast errors.

Use in thesis:

> Forecast errors can push predictive control below simple baseline performance, which motivates robustness-aware validation rather than relying on offline metrics alone.

#### Esrafilian-Najafabadi and Haghighat, 2022

Paraphrased abstract:

The paper compares different occupancy prediction formulations and machine learning techniques for HVAC control. It evaluates models not only by MAE or accuracy, but also by thermal comfort, energy efficiency, and a combined control-oriented performance score.

Key conclusions:

- Choosing the right prediction formulation can matter more than choosing the ML algorithm.
- MAE and accuracy showed only weak to moderate correlation with the overall control-performance score.
- Relying only on machine-learning metrics can fail to select the best model for HVAC control.

Use in thesis:

> In HVAC control, standard prediction metrics may correlate only weakly with control-oriented comfort-energy scores; therefore, predictor selection should include downstream control criteria.

#### Grant and Gehbauer, 2022

Paraphrased abstract:

The paper emulates weather forecast errors and evaluates their effects on MPC for dynamic facades. It finds that imperfect forecasts can increase cost and glare, while bias correction can remove most of the penalty in the studied case.

Key conclusions:

- Weather forecast error can degrade both economic and comfort-related outcomes.
- Bias correction can be more important than raw forecast model complexity.
- Forecast error should be evaluated through downstream cost and comfort measures.

Use in thesis:

> For MPC, correcting systematic forecast bias can be more valuable than reducing undifferentiated average error.

### E4. When prediction-control correlation is positive

Prediction accuracy is more likely to correlate positively with control performance when:

- model structure, controller, objective, constraints, and optimization algorithm are fixed
- the accuracy improvement occurs in variables that enter the control objective or constraints
- the improvement occurs in the first step or control horizon used by the receding-horizon update
- signed bias is reduced, not only variance
- input-output sensitivities and gradient signs are preserved
- prediction speed remains fast enough for online MPC
- the optimizer cannot exploit model artifacts

### E5. When prediction-control correlation becomes weak or negative

Generic prediction metrics can mislead MPC selection when:

- a model improves terminal or full-horizon MAE but worsens first-step or short-horizon error
- errors occur in a control-insensitive range, so offline metrics overstate their importance
- small errors occur in highly control-sensitive directions, so offline metrics understate their importance
- the model has correct values but wrong input-output sensitivity
- the model creates smooth but biased forecasts that drive systematic control errors
- active constraints depend on rare events diluted by average MAE
- exogenous forecast errors compound across several inputs
- the controller exploits unrealistic gradients or flat sensitivities
- computational delay changes the practical control policy

This is the situation most relevant to the current PHF result.

## F. Recommended Metrics For `agc_mpc`

### Forecast metrics aligned with receding-horizon MPC

| Metric | Definition | Why it matters |
| --- | --- | --- |
| First-step MAE | Error at horizon step 1. | The simulator currently advances the state using the first-step prediction. |
| Control-horizon MAE | Mean error over the first `N_c` steps, currently useful at `6` steps. | The first few steps dominate receding-horizon behavior. |
| Horizon-weighted MAE | Error with larger weights on control-relevant steps. | Better than treating all 24 steps equally. |
| Segment MAE | Separate early, middle, and late horizon errors. | Shows whether a model trades early accuracy for terminal accuracy. |
| Final-step MAE | Error at the terminal prediction step. | Still useful for long-horizon forecast claims. |
| Signed bias | Mean signed error per target and horizon segment. | Systematic over/under-prediction can be worse than zero-mean noise. |
| Constraint-near error | Error near constraint boundaries or target extremes. | MPC often cares most about these cases. |

### Control-sensitivity metrics

| Metric | Definition | Why it matters |
| --- | --- | --- |
| `dy/du` sensitivity | Change in predicted target caused by a small control-input perturbation. | Tests whether actuators have the right modeled effect. |
| Cost-gradient magnitude | Mean absolute gradient of the control objective with respect to future inputs. | Tests whether GradientMPC receives useful signal. |
| Gradient sign consistency | Whether sensitivity signs match physical/control expectations. | Wrong signs can make the optimizer act against the process. |
| Input-specific gradient share | Gradient magnitude by control input. | Identifies whether the model relies on plausible actuators. |
| Flatness / saturation score | How often gradients are near zero or clipped. | Explains inactive or unstable GradientMPC behavior. |

### Closed-loop validation metrics

| Metric | Definition | Why it matters |
| --- | --- | --- |
| MPC objective | The actual optimized/evaluated closed-loop cost. | Primary control metric. |
| Target MAE | Closed-loop MAE for `Tair`, `Rhair`, `CO2air`. | Separates overall objective from target-specific behavior. |
| Constraint violations | Count and severity of violations. | Necessary for control papers. |
| Resource/economic cost | Energy, CO2 dosing, ventilation, irrigation, or electricity cost when available. | Required for a mature greenhouse MPC story. |
| Action activity | Total variation or movement of control inputs. | Detects over-aggressive or inactive controllers. |
| Recorded-policy gap | Difference from observed greenhouse operation. | Useful while real cost data are incomplete. |
| Oracle gap | Difference from perfect-forecast or ideal-preview MPC. | Quantifies how much forecast error still matters. |
| Robustness under forecast perturbation | Closed-loop degradation under biased/noisy forecasts. | Turns robustness into an empirical claim. |

## G. How This Maps To Current Results

Current evidence:

- `itransformer_co2_horizon_mixture` is the offline CO2 forecasting leader:
  - CO2 Full MAE `43.910`
  - CO2 Final MAE `47.661`
- It is not the closed-loop MPC leader:
  - `GradientMPC` objective `0.3713`
  - closed-loop CO2 MAE `28.696`
- `itransformer_co2_late_frozen_expert` remains the strongest current CO2 control baseline:
  - closed-loop CO2 MAE `6.298`
- `itransformer_co2_recoupled_expert` remains the strongest current overall objective baseline:
  - objective `0.0651`
- Initial control-relevant validation ranks `late_residual`, `late_frozen_expert`, and `frozen_backbone_horizon_mixture` ahead of `horizon_mixture`.

Interpretation:

- The literature predicts exactly this kind of split.
- `horizon_mixture` improved full/final offline forecasting, but MPC currently relies heavily on first-step and short-horizon behavior.
- The paper story should not claim that the offline leader is automatically the control leader.
- The paper story should claim that PHF improves offline CO2 forecasting and reveals the need for control-relevant validation before MPC deployment.

## H. Suggested Thesis Paragraphs

### Forecasting architecture paragraph

Recent greenhouse forecasting studies show that greenhouse climate prediction is not a setting where plain Transformer models are automatically dominant. Competitive approaches include linear models, recurrent models with attention, hybrid ensembles, decomposition-based architectures, and variable-weight fusion. This motivates using a stable main predictor together with residual or specialist correction branches instead of replacing the whole system with a larger generic backbone.

### CO2 paragraph

Greenhouse CO2 forecasting is more regime-dependent than temperature forecasting because concentration dynamics are affected by dosing, ventilation exchange, crop uptake, and operating schedules. CO2-focused studies therefore often use decomposition, denoising, recurrent modeling, and adaptive fusion. This supports a CO2 specialist line and the PHF design, where a dedicated CO2 expert is integrated into a multi-target predictor through protected, horizon-aware correction.

### Control-relevant validation paragraph

Predictor selection for MPC cannot be based solely on generic open-loop forecasting metrics. Control-relevant identification and validation studies have shown that a model can be accurate in an ordinary fit sense while still being unsuitable for control, because the controller depends on multi-step predictions, input-output sensitivities, active constraints, and setpoint directions. Recent HVAC and energy-storage MPC studies similarly report that prediction metrics such as MAE, MSE, accuracy, or R2 may correlate with control performance within a fixed setup but can misrank models across different model structures or objectives. Therefore, this work evaluates greenhouse predictors using both offline forecasting metrics and control-relevant metrics, including first-step error, short-horizon error, horizon-weighted error, control sensitivity, and closed-loop MPC objective.

## I. Suggested Citation Roles

| Claim | Best citations |
| --- | --- |
| Greenhouse forecasting does not require plain Transformer dominance. | Ahn et al. 2024; Mao et al. 2024; Zeng et al. 2022 |
| Hybrid and variable-weight fusion are reasonable greenhouse forecasting strategies. | Mao et al. 2024; OneNet 2023; TimeMixer 2024 |
| CO2 benefits from decomposition and adaptive fusion. | Multi-model CO2 fusion 2024; wavelet-decoupled GRU 2025; mushroom CO2 optimized LSTM 2025 |
| CO2 should eventually be connected to carbon balance and crop uptake. | Acock et al. 1991; Nederhoff and Vegter 1994; model-based CO2 control 2007 |
| Control-relevant validation is a known concept. | Huang et al. 2003; Potts et al. 2014 |
| Ordinary accuracy is not sufficient for multivariable control. | Misra et al. 2017 |
| Model mismatch impact is direction-dependent and not always monotonic. | Badwe et al. 2010 |
| Neural MPC models should be trained/evaluated for their predictive-control role. | Lawrynczuk 2010 |
| Forecast MSE can misrank MPC value in modern energy systems. | Ludolfinger et al. 2025 |
| Prediction accuracy can still help under fixed setup. | Wang et al. 2024 |
| Forecast correction and bias matter for MPC. | Hou et al. 2022; Grant and Gehbauer 2022 |
| Occupancy/HVAC prediction metrics can weakly correlate with control scores. | Jain et al. 2018; Esrafilian-Najafabadi and Haghighat 2022 |
| Mature greenhouse MPC should include uncertainty/economic/resource terms. | Chen and You 2022; Kim and You 2025; Svensen et al. 2024 |

## J. Next Literature Tasks

1. Add full bibliographic metadata for every paper that becomes a final thesis citation.
2. Add greenhouse-specific MPC papers that explicitly compare forecast quality and control performance.
3. Add papers on economic MPC, stochastic MPC, and robust MPC metrics.
4. Add differentiable MPC or neural surrogate gradient-quality literature.
5. Build final thesis tables separating:
   - pure forecasting metrics
   - control-relevant validation metrics
   - closed-loop control metrics
   - resource/economic metrics
