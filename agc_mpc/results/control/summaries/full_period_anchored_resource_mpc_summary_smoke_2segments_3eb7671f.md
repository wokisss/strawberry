# Full-Period Anchored Resource MPC

This table summarizes repeated anchored closed-loop MPC segments over the Reference test split. Each segment is re-anchored to true observed AGC history before optimizing the next control window.

| predictor | w | segments | period | objective | CO2 MAE | heat | electricity | CO2 use | irrigation | resource cost | cost vs w=0 | CO2 vs w=0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_hybrid_transformer | 0.00 | 2 | 2020-05-06T06:25:00.000000000 to 2020-05-06T10:20:00.000000000 | 0.0474 | 17.655 | 0.220 | 0.015 | 0.0098 | 0.759 | 0.0037 | 0.0% | 0.0% |
| current_hybrid_transformer | 0.05 | 2 | 2020-05-06T06:25:00.000000000 to 2020-05-06T10:20:00.000000000 | 0.0679 | 17.699 | 0.199 | 0.009 | 0.0097 | 0.746 | 0.0031 | -15.8% | 0.3% |
| itransformer_co2_residual | 0.00 | 2 | 2020-05-06T06:25:00.000000000 to 2020-05-06T10:20:00.000000000 | 0.0385 | 2.895 | 0.180 | 0.011 | 0.0098 | 0.758 | 0.0031 | 0.0% | 0.0% |
| itransformer_co2_residual | 0.05 | 2 | 2020-05-06T06:25:00.000000000 to 2020-05-06T10:20:00.000000000 | 0.0555 | 3.196 | 0.175 | 0.007 | 0.0096 | 0.751 | 0.0028 | -10.4% | 10.4% |

Boundary:

- MPC resource values are counterfactual estimates from the calibrated AGC resource estimator.
- The comparison supports resource-cost and climate-control trade-off claims only.
- It does not claim true net-profit, yield, or quality improvement.

Segment records: `8`.