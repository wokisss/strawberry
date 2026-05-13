# Mainline Real-Resource Control Comparison

Selected closed-loop MPC rollouts are evaluated with the calibrated AGC resource estimator. Values are estimated for the 96-step rollout window, not for a season-long greenhouse crop.

| profile | predictor | starts | objective | CO2 MAE | heat | electricity | CO2 use | irrigation | resource cost |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real_resource_w000 | current_hybrid_transformer | 0,96,192,288,384 | 0.0660 | 29.472 | 0.283 | 0.115 | 0.0271 | 2.162 | 0.0127 |
| real_resource_w000 | itransformer_co2_residual | 0,96,192,288,384 | 0.0695 | 10.168 | 0.044 | 0.105 | 0.0265 | 2.008 | 0.0094 |
| real_resource_w005 | current_hybrid_transformer | 0,96,192,288,384 | 0.0841 | 29.929 | 0.235 | 0.098 | 0.0269 | 2.124 | 0.0114 |
| real_resource_w005 | itransformer_co2_residual | 0,96,192,288,384 | 0.0879 | 10.980 | 0.041 | 0.094 | 0.0262 | 2.002 | 0.0085 |

Boundary:

- This comparison uses model-generated action trajectories and a calibrated resource estimator.
- It is valid for tracking/resource trade-off analysis of selected MPC rollouts.
- It does not claim true commercial net-profit improvement because production and quality dynamics are not part of the closed-loop surrogate.

Detail records: `20` rollouts.