# Protected Horizon Fusion Ablation Summary

Lower MAE/objective/rank values are better. Blank control cells mean the model has no recorded closed-loop summary.

| label | role | co2air_full_mae | co2air_final_mae | gradient_objective | gradient_co2_mae | control_relevant_mean_rank | question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Residual | generic residual backbone baseline | 51.161 | 52.014 | 0.192 | 11.532 | 4.250 | Generic residual backbone |
| Late residual | CO2-aware backbone | 47.797 | 50.139 | 0.071 | 10.125 | 3.375 | Does a CO2-aware late adapter help? |
| Frozen expert | naive frozen-expert fusion baseline | 46.966 | 59.247 |  |  |  | Does a frozen standalone expert help if blended directly? |
| Late frozen expert | late-trust control baseline | 44.727 | 57.193 | 0.153 | 6.298 | 2.875 | Is horizon-dependent late trust useful? |
| Teacher distill | distillation ablation | 56.018 | 57.294 |  |  |  | Is using the expert only as a teacher enough? |
| Recoupled expert | cross-target recoupling baseline | 47.585 | 58.054 | 0.065 | 16.749 | 6.000 | Does recoupling after expert correction improve control objective? |
| Protected expert | agreement-protection ablation | 45.190 | 55.984 |  |  |  | Is agreement protection useful? |
| Protected terminal | terminal-loss ablation | 48.055 | 52.056 |  |  |  | Is terminal loss alone enough? |
| Horizon mixture | proposed offline PHF representative | 43.910 | 47.661 | 0.371 | 28.696 | 5.500 | Does protected horizon fusion with terminal pullback improve offline CO2? |
| Frozen-backbone mix | control-safety diagnostic | 46.334 | 50.139 | 0.072 | 10.000 | 4.250 | Does freezing the backbone improve MPC safety? |
| Control-aware fusion | late-frozen anchor + PHF terminal candidate | 43.983 | 49.069 | 0.149 | 6.415 | 1.750 | Can we keep late-frozen control behavior while recovering PHF terminal gains? |

## Main Reading

- `Horizon mixture` is the current offline PHF representative and CO2 forecasting leader.
- `Late frozen expert` remains the strongest current closed-loop CO2 control baseline.
- `Control-aware fusion` is the new single-candidate follow-up that should be judged by both control-relevant rank and `GradientMPC` transfer, not offline CO2 alone.
- `Recoupled expert` remains the strongest current overall closed-loop objective baseline.
- `Frozen-backbone mix` is a control-safety diagnostic, not the main offline forecasting method.
