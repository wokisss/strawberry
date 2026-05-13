# AGC Resource Cost Model Coefficient Diagnosis

The calibrated resource-cost estimator is intentionally simple: daily setpoint/weather summaries are mapped to recorded daily resource use with positive-coefficient ridge regression. The model is used as a bridge from MPC action trajectories to estimated resource implications, not as a physical greenhouse simulator.

## Validation Quality

| target | MAE | R2 | interpretation |
| --- | ---: | ---: | --- |
| heat consumption | 0.5657 | 0.620 | usable for coarse heat-cost comparison |
| electricity consumption | 0.2816 | 0.835 | strongest estimator; suitable for comparison |
| CO2 consumption | 0.0102 | 0.731 | suitable for comparison |
| irrigation | 1.0140 | 0.448 | weak; use only as auxiliary context |

## Coefficient Pattern

Heat consumption is driven mainly by seasonal/weather terms and the derived heat drive. This is plausible because heating demand depends strongly on outside conditions and the gap between heating setpoint and outside temperature.

Electricity consumption is explained strongly by seasonal/time terms and lighting-related controls. This is plausible because artificial lighting and photoperiod strategy dominate electricity demand in the AGC data.

CO2 consumption is dominated by seasonal/time terms and weather leakage proxies. This is partly plausible but also shows that the recorded CO2 use is not explained cleanly by `co2_sp` alone. The estimator should therefore be treated as empirical and dataset-calibrated.

Irrigation has the weakest fit and is dominated by seasonal/time terms plus some ventilation/window pressure. This means irrigation estimates are useful for qualitative comparison only, not as a strong claim.

## Practical Boundary

- The estimator is adequate for selected-model resource-cost comparison over short MPC rollouts.
- The estimator is not adequate for season-long yield, quality, or net-profit prediction.
- The thesis should describe it as a real-AGC-resource-calibrated cost estimator, not as a mechanistic greenhouse economics model.
