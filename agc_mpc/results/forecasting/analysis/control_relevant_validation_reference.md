# Control-Relevant Validation Summary

Lower ranks and lower MAE/objective values are better. Gradient columns are diagnostic magnitudes, not direct objectives.

| predictor | control_relevant_mean_rank | co2_first_step_mae | co2_control_horizon_mae | co2_control_horizon_bias | co2_constraint_near_mae_proxy | co2_final_step_mae | mpc_objective | mpc_co2_mae | recorded_policy_co2_improvement | cost_grad_mean_abs | co2_sp_first_grad_signed | co2_sp_first_grad_positive_fraction | co2_sp_first_grad | t_vent_sp_first_grad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| itransformer_co2_control_aware_fusion | 1.7500 | 24.4675 | 26.7417 | -3.6691 | 29.3921 | 26.6010 | 0.1491 | 6.4151 | 39.6354 | 0.0154 | 0.3040 | 0.1667 | 0.3040 | 0.1893 |
| itransformer_co2_late_frozen_expert | 2.8750 | 24.4675 | 26.7417 | -3.6691 | 31.3419 | 63.8859 | 0.1533 | 6.2976 | 39.7529 | 0.0154 | 0.3040 | 0.1667 | 0.3040 | 0.1893 |
| itransformer_co2_late_residual | 3.3750 | 25.4535 | 30.6503 | 18.6592 | 32.0419 | 29.3823 | 0.0705 | 10.1250 | 44.0709 | 0.0183 | 0.1952 | 0.1667 | 0.1952 | 0.2199 |
| itransformer_residual | 4.2500 | 29.8241 | 32.6112 | 12.8798 | 33.7574 | 27.7324 | 0.1924 | 11.5319 | 33.0101 | 0.0169 | 0.3494 | 0.1667 | 0.3494 | 0.1535 |
| itransformer_co2_frozen_backbone_horizon_mixture | 4.2500 | 25.4535 | 30.7720 | 18.9436 | 34.5681 | 29.3823 | 0.0718 | 9.9996 | 44.1964 | 0.0196 | 0.1952 | 0.1667 | 0.1952 | 0.2199 |
| itransformer_co2_horizon_mixture | 5.5000 | 32.7845 | 35.4709 | 15.5567 | 39.2123 | 26.8922 | 0.3713 | 28.6959 | 17.9707 | 0.0194 | 0.2156 | 0.1667 | 0.2156 | 0.1738 |
| itransformer_co2_recoupled_expert | 6.0000 | 35.2999 | 41.9332 | 37.0571 | 45.8024 | 59.6514 | 0.0651 | 16.7488 | 37.8732 | 0.0171 | 0.0657 | 0.1667 | 0.0657 | 0.0378 |
