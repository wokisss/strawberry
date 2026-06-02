# Full-Period Anchored Resource MPC

This table summarizes repeated anchored closed-loop MPC segments over the Reference test split. Each segment is re-anchored to true observed AGC history before optimizing the next control window.

| predictor | w | segments | period | objective | CO2 MAE | heat | electricity | CO2 use | irrigation | resource cost | cost vs w=0 | CO2 vs w=0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_hybrid_transformer | 0.00 | 283 | 2020-05-06T06:25:00.000000000 to 2020-05-29T20:20:00.000000000 | 0.0632 | 27.659 | 12.106 | 1.963 | 1.2825 | 109.675 | 0.3455 | 0.0% | 0.0% |
| current_hybrid_transformer | 0.05 | 283 | 2020-05-06T06:25:00.000000000 to 2020-05-29T20:20:00.000000000 | 0.0794 | 27.811 | 10.660 | 1.792 | 1.2743 | 109.253 | 0.3215 | -6.9% | 0.6% |
| itransformer_co2_residual | 0.00 | 283 | 2020-05-06T06:25:00.000000000 to 2020-05-29T20:20:00.000000000 | 0.0710 | 15.112 | 12.832 | 1.909 | 1.3286 | 109.610 | 0.3423 | 0.0% | 0.0% |
| itransformer_co2_residual | 0.05 | 283 | 2020-05-06T06:25:00.000000000 to 2020-05-29T20:20:00.000000000 | 0.0897 | 15.706 | 11.918 | 1.611 | 1.3060 | 108.723 | 0.3133 | -8.5% | 3.9% |

Boundary:

- MPC resource values are counterfactual estimates from the calibrated AGC resource estimator.
- The comparison supports resource-cost and climate-control trade-off claims only.
- It does not claim true net-profit, yield, or quality improvement.

Segment records: `1132`.