# CO2_SPECIALIST_REPORT.md

English canonical version.
Mapped Chinese mirror: [CO2_SPECIALIST_REPORT.zh-CN.md](c:/repositories/strawberry/agc_mpc/CO2_SPECIALIST_REPORT.zh-CN.md)
Last synchronized: `2026-04-07`

## 1. What This Report Covers

This report focuses on one question:

Among the papers listed in [CO2_PAPERS_AND_DIRECTION.md](c:/repositories/strawberry/agc_mpc/CO2_PAPERS_AND_DIRECTION.md), which direct greenhouse `CO2` forecasting methods are worth implementing first, what has already been implemented, how well it works, and how it should be merged back into the current `agc_mpc` mainline.

Here, "implemented" means:

- converted into runnable models under the current `AGC` data interface
- trained under the formal fair-budget benchmark
- summarized with architecture, principle, transfer value, and next-step priority

## 2. Executive Conclusion

Three standalone `CO2air` specialist lines have already been implemented:

1. `co2_env_lstm`
2. `co2_vmd_lstm_fusion`
3. `co2_wavelet_gru_attn`

Current fair-budget ranking:

1. `co2_wavelet_gru_attn`
   - Full `R2 = 0.7519`
   - Full `MAE = 45.209`
   - Final `R2 = 0.6159`
   - Final `MAE = 58.292`
2. `co2_vmd_lstm_fusion`
   - Full `R2 = 0.6863`
   - Full `MAE = 52.298`
   - Final `R2 = 0.6003`
   - Final `MAE = 59.697`
3. `co2_env_lstm`
   - Full `R2 = 0.3065`
   - Full `MAE = 74.157`
   - Final `R2 = -0.4852`
   - Final `MAE = 118.800`

Direct interpretation:

- A plain "environmental factors + LSTM" line is not enough.
- `CO2` benefits from multi-scale decomposition and adaptive fusion.
- This matches the literature.
- The best current standalone direction is `wavelet / multi-scale + GRU + adaptive attention`.

## 3. Implemented Files

### 3.1 Model File

- [co2_specialist_forecasters.py](c:/repositories/strawberry/agc_mpc/models/co2_specialist_forecasters.py)

Implemented models:

- `ConditionalCO2LSTMForecaster`
- `ConditionalCO2VMDLSTMFusionForecaster`
- `ConditionalCO2WaveletGRUAttnForecaster`

### 3.2 Benchmark Entry

- [benchmark_co2_specialist_forecasters.py](c:/repositories/strawberry/agc_mpc/benchmark_co2_specialist_forecasters.py)

Protocol:

- dataset: `AGC`
- regime: `joint_all + Reference eval`
- target: only `CO2air`
- budget: `batch_size=256`, `epochs=200`, `lr=1e-4`, `patience=15`

### 3.3 Plotting Entry

- [plot_co2_specialist_forecasters.py](c:/repositories/strawberry/agc_mpc/plot_co2_specialist_forecasters.py)

Figure output directory:

- [results/forecasting/figures/co2_specialists](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/co2_specialists)

## 4. Paper-To-Model Mapping

The paper mapping should be read at two levels:

1. what the original paper is really trying to do
2. how we translated that idea into a runnable structure inside the current repository

Important clarification:

- the implementations below are paper-inspired engineering translations
- they are not full paper reproductions
- modules such as `WT`, `VMD`, `SSA`, `DBO`, or exact wavelet toolchains are approximated in a way that fits the current training stack

## 5. Paper-By-Paper Briefing

### 5.1 Prediction of CO2 Concentration via Long Short-Term Memory Using Environmental Factors in Greenhouses

Sources:

- [Horticultural Science and Technology / DOI 10.7235/HORT.20200019](https://www.hst-j.org/articles/xml/ozK9/)
- [KCI record](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002578287)

What the paper does:

- It directly predicts greenhouse `CO2`.
- It does not treat `CO2` only as one output among many.
- The setting is a mango greenhouse.
- Inputs include temperature, humidity, solar radiation, pressure, soil temperature, soil moisture, and historical `CO2`.
- Output is future `CO2` up to `2 h` ahead.

Core idea:

- `CO2` is a task worth modeling on its own.
- A recurrent model such as `LSTM` can absorb delayed effects from environment and greenhouse regime.
- Historical `CO2` itself is one of the strongest signals.

High-level architecture:

1. feed a historical environmental sequence
2. encode temporal dependence with `LSTM`
3. decode future `CO2`

What this paper really teaches us:

- the important lesson is not "LSTM is unbeatable". The important lesson is that `CO2` deserves a dedicated model path.
- if control signals or operation logs are missing, a model can easily underfit the dosing-driven peaks

Repository implementation:

- mapped model: `co2_env_lstm`
- file: [co2_specialist_forecasters.py](c:/repositories/strawberry/agc_mpc/models/co2_specialist_forecasters.py)

Implemented structure:

1. `x_past` enters an `LSTM` encoder
2. `w_future + u_future` enter a future-conditioning embedding
3. a decoder `LSTM` produces future hidden states
4. the model predicts a `CO2` increment on top of the last observed `CO2`

Why the last-observation anchor matters:

- directly regressing absolute `CO2` trajectories was unstable
- `CO2` is strongly autoregressive
- the last observation is a necessary anchor
- this is consistent with the spirit of autoregressive and `NARX`-style modeling

How to interpret current performance:

- it is the weakest of the three standalone CO2 lines
- this means a single plain recurrent backbone is not enough for greenhouse `CO2`
- but it is still valuable as a clean specialist baseline

What we can borrow:

1. single-target `CO2` modeling is justified
2. an autoregressive anchor should be preserved
3. a pure `LSTM` line is useful as a teacher, baseline, or ablation

How to integrate it into the mainline:

- do not replace the main multi-target forecaster with it
- use it as a clean CO2 specialist baseline or auxiliary expert

Priority:

- Medium

### 5.2 Time-serial analysis of deep neural network models for prediction of climatic conditions inside a greenhouse

Sources:

- [ScienceDirect / DOI 10.1016/j.compag.2020.105402](https://www.sciencedirect.com/science/article/pii/S0168169919317326)
- [KIST abstract page](https://pubs.kist.re.kr/handle/201004/118578)

What the paper does:

- It does not only predict `CO2`.
- It compares temperature, humidity, and `CO2` together.
- It benchmarks `ANN`, `NARX`, and `RNN-LSTM`.
- The focus is how they behave across time-serial prediction settings.

Most important conclusion:

- In greenhouse systems with strong temporal lag, plain feedforward `ANN` is not enough.
- `NARX` and `RNN-LSTM` are better matched to the dynamics.
- `RNN-LSTM` is the most stable model family in the study.
- `CO2` is clearly harder than temperature.

Why this matters:

- it says recurrent memory still matters in greenhouse forecasting
- it warns against assuming that a larger generic architecture is automatically better
- it confirms that `CO2` deserves stronger temporal treatment than easy variables

How we borrowed it:

- we did not reproduce the full `ANN / NARX / RNN-LSTM` comparison stack
- instead, we concentrated its practical lesson into the standalone `co2_env_lstm` line
- the point was to answer: how far can a clean recurrent CO2-only baseline go?

What we can borrow further:

1. keep a recurrent branch inside CO2 modeling
2. add more explicit autoregressive inputs if needed
3. do not discard recurrent specialists too early

Priority:

- High as structural evidence, but not necessarily as the final strongest architecture

### 5.3 Multi-model fusion method for predicting CO2 concentration in greenhouse tomatoes

Source:

- [ScienceDirect / DOI 10.1016/j.compag.2024.109623](https://www.sciencedirect.com/science/article/pii/S0168169924010147)

What the paper does:

- It directly predicts greenhouse tomato `CO2`.
- It explicitly argues that a single model is not enough for a non-stationary and noisy `CO2` series.
- It therefore uses a "decompose, model, then fuse" route.

Stable structure that can be extracted from the abstract and highlights:

1. `WT` for denoising
2. `VMD` for multi-scale decomposition
3. `LSTM` to model decomposed components
4. `attention` to emphasize important temporal content
5. final fusion into a `CO2` prediction

Core principle:

- `CO2` is not a single-scale variable
- it mixes slow diurnal trends, medium-scale ventilation or dosing changes, and sharp local disturbances
- those bands should be modeled separately and fused adaptively

Why this is directly relevant to us:

- we already observe that `CO2air` can look acceptable on average metrics while still drifting badly on rollout windows
- that is exactly the kind of mixed-scale failure this paper is addressing

Repository implementation:

- mapped model: `co2_vmd_lstm_fusion`

Implemented structure:

1. approximate trend/detail decomposition using smoothing filters
2. feed trend and detail into separate `LSTM` encoders
3. turn future weather and control into query tokens
4. attend to both branches separately
5. fuse branches with a dynamic gate
6. predict a `CO2` increment on top of the last observed `CO2`

Why this is a reasonable paper-inspired translation:

- the repository does not currently carry a full `WT + VMD` stack
- but the key skeleton, namely `decomposition + LSTM + attention + fusion`, is preserved
- this is the most practical engineering approximation for now

How to interpret current performance:

- it is clearly better than the plain `LSTM`
- so decomposition and fusion are doing useful work
- but it is not yet the strongest standalone line

What we can borrow:

1. keep multi-scale decomposition
2. avoid using a single encoder for all CO2 modes
3. use dynamic branch fusion rather than fixed weighting

How to integrate it into the mainline:

- best fit: a `CO2 residual expert` inside the current multi-target predictor
- not a full replacement for the whole multi-target model

Priority:

- Very High

### 5.4 Prediction of CO2 concentration in mushroom greenhouse via optimized long and short term memory algorithm

Sources:

- [Scientific Reports / DOI 10.1038/s41598-025-86394-0](https://www.nature.com/articles/s41598-025-86394-0)
- [PMC open version](https://pmc.ncbi.nlm.nih.gov/articles/PMC12485007/)

What the paper does:

- It predicts `CO2` in a mushroom greenhouse.
- It does not only change the backbone.
- It combines decomposition and optimization.
- Compared models include:
  - `LSTM`
  - `EMD-LSTM`
  - `VMD-LSTM`
  - `VMD-SSA-LSTM`
  - `VMD-DBO-LSTM`

Very clear modeling chain:

1. decompose the `CO2` series into multiple components
2. let `LSTM` model the decomposed components
3. use optimization algorithms such as `SSA` or `DBO` to search better hyperparameters
4. obtain a more accurate forecaster

Two-layer lesson from this paper:

First:

- `VMD`-style decomposition helps `CO2`

Second:

- performance does not come only from the backbone
- it also comes from getting the decomposition and hyperparameters right

Why this matters:

- it warns us not to obsess only over architecture swapping
- `CO2` is likely sensitive to hidden size, learning rate, decomposition granularity, and horizon weighting

How it is reflected in the current repository:

- it is one of the main conceptual supports behind `co2_vmd_lstm_fusion`
- we have not yet implemented `SSA` or `DBO` search directly

What we can borrow next:

1. add structured hyperparameter search to the decomposition-fusion line
2. focus the search on:
   - decomposition granularity
   - hidden size
   - learning rate
   - horizon-aware loss weighting
3. keep it as offline search first rather than putting it inside the main training loop

Priority:

- High

### 5.5 Wavelet-decoupled GRU with adaptive attention for multi-step carbon dioxide concentration prediction in intelligent glass greenhouse

Source:

- [ScienceDirect / DOI 10.1016/j.atech.2025.101653](https://www.sciencedirect.com/science/article/pii/S2772375525008846)

What the paper does:

- This is the paper that is currently closest to our problem.
- It is designed specifically for multi-step greenhouse `CO2` forecasting.
- It directly targets the accumulation of error over longer horizons.

The abstract and highlights give a clear architecture outline:

1. wavelet or frequency decoupling at the front
2. `GRU` over decomposed multi-scale features
3. position-adjustable multi-head attention for multi-step forecasting
4. explicit attention to `1 h / 2 h / 4 h / 8 h` forecasting conditions

Core principle:

- long-horizon `CO2` forecasting fails not just because the model is weak
- it fails because:
  - slow cycles and sharp disturbances are mixed together
  - the importance of each band changes with horizon
- therefore the model should:
  - separate temporal bands first
  - then change fusion weights with horizon

Why this is currently the best fit for us:

- our current problem is exactly that `CO2air` degrades on later forecast steps
- this paper is about multi-step windows, not just point prediction
- it explicitly treats late-horizon behavior as different from early-horizon behavior

Repository implementation:

- mapped model: `co2_wavelet_gru_attn`

Implemented structure:

1. approximate `low / mid / high` temporal bands using smoothing filters
2. feed each band into its own `GRU` encoder
3. build future query tokens from weather, control, and horizon ratio
4. attend to all three bands separately
5. use a softmax-based adaptive fusion weight
6. predict a `CO2` increment on top of the last observed `CO2`

Why this line is currently strongest:

- it captures both "multi-scale" and "horizon-aware" behavior.
- those are exactly the two things `CO2` seems to need most in our setting

How to interpret current performance:

- it is the best of the three standalone CO2 models
- Full `MAE = 45.209`, which is already better than the current multi-target `itransformer_co2_late_residual` Full `MAE = 47.797`
- but the Final `MAE = 58.292` is still not ideal, which means the tail end of the horizon still needs work

What we can borrow:

1. keep explicit multi-scale band branches
2. keep `GRU` as a serious CO2 backbone candidate
3. make the fusion weight explicitly horizon-dependent

How to integrate it into the mainline:

- this is the highest-priority idea to merge back into the multi-target CO2 residual line
- the natural path is to replace the current CO2 adapter with a wavelet-inspired multi-scale specialist

Priority:

- Very High, currently the top priority

## 6. What Should Be Implemented Next, In Batches

### Batch 1: Already Implemented

Goal:

- translate the most relevant direct greenhouse `CO2` forecasting methods into runnable code

Completed:

1. `co2_env_lstm`
2. `co2_vmd_lstm_fusion`
3. `co2_wavelet_gru_attn`

### Batch 2: Highest Priority Next Step

Goal:

- merge the strongest standalone CO2 idea back into the multi-target mainline

Recommended order:

1. use `co2_wavelet_gru_attn` as the template for a new multi-target CO2 residual branch
2. apply the specialist correction only to the `CO2air` channel
3. make the fusion explicitly horizon-aware
4. validate whether `GradientMPC` also benefits on the control benchmark

### Batch 3: High Priority

Goal:

- bring in the paper lesson that hyperparameter optimization matters for `CO2`

Recommended approach:

1. do not start with a full `SSA / DBO` implementation immediately
2. first add lightweight automated search over:
   - hidden size
   - decomposition kernel / granularity
   - attention heads
   - horizon-weighted loss
3. only add more formal search machinery if the gain is real

### Batch 4: Research Upgrade

Goal:

- move from a pure single-target black-box predictor toward carbon-balance gray-box modeling

Directions:

1. `CO2 dosing`
2. ventilation exchange
3. canopy uptake / photosynthesis
4. respiration

This is not the first next step, because the more urgent need is still to improve forecasting strength and closed-loop transfer.

## 7. Most Report-Worthy Current Conclusion

The cleanest weekly-report version is:

- We have already converted the most useful greenhouse `CO2` forecasting ideas from the literature into three standalone benchmark lines.
- The results show that a plain `LSTM` is not enough, and that `CO2` benefits more from multi-scale decomposition and adaptive fusion.
- The strongest current method is `wavelet-inspired + GRU + adaptive attention`.
- That line reaches Full `CO2air MAE = 45.209` under the current fair-budget AGC benchmark.
- Therefore, the next step should be to merge this standalone CO2 specialist logic back into the current multi-target mainline, rather than continuing to swap generic backbones.
## 8. 2026-04-07 Multi-Target Integration Note

After the standalone CO2 specialist benchmark, two direct multi-target integration attempts were tested:

1. `itransformer_co2_wavelet_residual`
2. `itransformer_co2_wavelet_blend`

Formal results:

- `itransformer_co2_wavelet_residual`
  - `CO2air`: Full `R2=0.5182`, MAE `65.984`
- `itransformer_co2_wavelet_blend`
  - `CO2air`: Full `R2=0.5813`, MAE `64.666`

These are both worse than:

- `itransformer_residual`: Full `CO2air MAE = 51.161`
- `itransformer_co2_late_residual`: Full `CO2air MAE = 47.797`
- standalone `co2_wavelet_gru_attn`: Full `CO2air MAE = 45.209`

Current takeaway:

- the standalone specialist is strong
- but naive end-to-end integration into the multi-target model breaks its advantage
- the next more reasonable route is not another immediate branch rewrite, but a more decoupled transfer method such as frozen-expert fusion, distillation, or teacher-guided auxiliary loss