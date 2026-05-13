# AGC Resource Cost Model Validation

The estimator is a simple positive-coefficient ridge model fitted on daily AGC records. It is intended as an interpretable cost bridge from MPC action trajectories to approximate resource implications, not as a crop-profit simulator.

Serialized model spec: `agc_resource_cost_model.json`

| target | samples | MAE | R2 | observed mean | predicted mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| heat_cons_mj_m2 | 996 | 0.5657 | 0.620 | 1.7867 | 1.7867 |
| electricity_kwh_m2 | 996 | 0.2816 | 0.835 | 1.6204 | 1.6204 |
| co2_cons_kg_m2 | 996 | 0.0102 | 0.731 | 0.0575 | 0.0575 |
| irrigation_l_m2 | 996 | 1.0140 | 0.448 | 4.4797 | 4.4797 |

Boundary:

- The model estimates recorded daily resource consumption from setpoint/weather summaries.
- It can compare selected short MPC rollouts after scaling the estimate by rollout length.
- It must not be used to claim season-long commercial net-profit improvement, because yield dynamics are not modeled in closed-loop.