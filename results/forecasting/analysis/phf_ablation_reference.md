# Protected Horizon Fusion Ablation Summary

Lower MAE/objective/rank values are better. Blank control cells mean the model has no recorded closed-loop summary.

| label | role | co2air_full_mae | co2air_final_mae | gradient_objective | gradient_co2_mae | control_relevant_mean_rank | question |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Residual | generic residual backbone baseline |  |  |  |  |  | Generic residual backbone |

## Main Reading

- `Horizon mixture` is the current offline PHF representative and CO2 forecasting leader.
- `Late frozen expert` remains the strongest current closed-loop CO2 control baseline.
- `Control-aware fusion` is the new single-candidate follow-up that should be judged by both control-relevant rank and `GradientMPC` transfer, not offline CO2 alone.
- `Recoupled expert` remains the strongest current overall closed-loop objective baseline.
- `Frozen-backbone mix` is a control-safety diagnostic, not the main offline forecasting method.
| Late residual | CO2-aware backbone |  |  |  |  |  | Does a CO2-aware late adapter help? |

## Main Reading

- `Horizon mixture` is the current offline PHF representative and CO2 forecasting leader.
- `Late frozen expert` remains the strongest current closed-loop CO2 control baseline.
- `Control-aware fusion` is the new single-candidate follow-up that should be judged by both control-relevant rank and `GradientMPC` transfer, not offline CO2 alone.
- `Recoupled expert` remains the strongest current overall closed-loop objective baseline.
- `Frozen-backbone mix` is a control-safety diagnostic, not the main offline forecasting method.
| Frozen expert | naive frozen-expert fusion baseline |  |  |  |  |  | Does a frozen standalone expert help if blended directly? |

## Main Reading

- `Horizon mixture` is the current offline PHF representative and CO2 forecasting leader.
- `Late frozen expert` remains the strongest current closed-loop CO2 control baseline.
- `Control-aware fusion` is the new single-candidate follow-up that should be judged by both control-relevant rank and `GradientMPC` transfer, not offline CO2 alone.
- `Recoupled expert` remains the strongest current overall closed-loop objective baseline.
- `Frozen-backbone mix` is a control-safety diagnostic, not the main offline forecasting method.
| Late frozen expert | late-trust control baseline |  |  |  |  |  | Is horizon-dependent late trust useful? |

## Main Reading

- `Horizon mixture` is the current offline PHF representative and CO2 forecasting leader.
- `Late frozen expert` remains the strongest current closed-loop CO2 control baseline.
- `Control-aware fusion` is the new single-candidate follow-up that should be judged by both control-relevant rank and `GradientMPC` transfer, not offline CO2 alone.
- `Recoupled expert` remains the strongest current overall closed-loop objective baseline.
- `Frozen-backbone mix` is a control-safety diagnostic, not the main offline forecasting method.
| Teacher distill | distillation ablation |  |  |  |  |  | Is using the expert only as a teacher enough? |

## Main Reading

- `Horizon mixture` is the current offline PHF representative and CO2 forecasting leader.
- `Late frozen expert` remains the strongest current closed-loop CO2 control baseline.
- `Control-aware fusion` is the new single-candidate follow-up that should be judged by both control-relevant rank and `GradientMPC` transfer, not offline CO2 alone.
- `Recoupled expert` remains the strongest current overall closed-loop objective baseline.
- `Frozen-backbone mix` is a control-safety diagnostic, not the main offline forecasting method.
| Recoupled expert | cross-target recoupling baseline |  |  |  |  |  | Does recoupling after expert correction improve control objective? |

## Main Reading

- `Horizon mixture` is the current offline PHF representative and CO2 forecasting leader.
- `Late frozen expert` remains the strongest current closed-loop CO2 control baseline.
- `Control-aware fusion` is the new single-candidate follow-up that should be judged by both control-relevant rank and `GradientMPC` transfer, not offline CO2 alone.
- `Recoupled expert` remains the strongest current overall closed-loop objective baseline.
- `Frozen-backbone mix` is a control-safety diagnostic, not the main offline forecasting method.
| Protected expert | agreement-protection ablation |  |  |  |  |  | Is agreement protection useful? |

## Main Reading

- `Horizon mixture` is the current offline PHF representative and CO2 forecasting leader.
- `Late frozen expert` remains the strongest current closed-loop CO2 control baseline.
- `Control-aware fusion` is the new single-candidate follow-up that should be judged by both control-relevant rank and `GradientMPC` transfer, not offline CO2 alone.
- `Recoupled expert` remains the strongest current overall closed-loop objective baseline.
- `Frozen-backbone mix` is a control-safety diagnostic, not the main offline forecasting method.
| Protected terminal | terminal-loss ablation |  |  |  |  |  | Is terminal loss alone enough? |

## Main Reading

- `Horizon mixture` is the current offline PHF representative and CO2 forecasting leader.
- `Late frozen expert` remains the strongest current closed-loop CO2 control baseline.
- `Control-aware fusion` is the new single-candidate follow-up that should be judged by both control-relevant rank and `GradientMPC` transfer, not offline CO2 alone.
- `Recoupled expert` remains the strongest current overall closed-loop objective baseline.
- `Frozen-backbone mix` is a control-safety diagnostic, not the main offline forecasting method.
| Horizon mixture | proposed offline PHF representative |  |  |  |  |  | Does protected horizon fusion with terminal pullback improve offline CO2? |

## Main Reading

- `Horizon mixture` is the current offline PHF representative and CO2 forecasting leader.
- `Late frozen expert` remains the strongest current closed-loop CO2 control baseline.
- `Control-aware fusion` is the new single-candidate follow-up that should be judged by both control-relevant rank and `GradientMPC` transfer, not offline CO2 alone.
- `Recoupled expert` remains the strongest current overall closed-loop objective baseline.
- `Frozen-backbone mix` is a control-safety diagnostic, not the main offline forecasting method.
| Frozen-backbone mix | control-safety diagnostic |  |  |  |  |  | Does freezing the backbone improve MPC safety? |

## Main Reading

- `Horizon mixture` is the current offline PHF representative and CO2 forecasting leader.
- `Late frozen expert` remains the strongest current closed-loop CO2 control baseline.
- `Control-aware fusion` is the new single-candidate follow-up that should be judged by both control-relevant rank and `GradientMPC` transfer, not offline CO2 alone.
- `Recoupled expert` remains the strongest current overall closed-loop objective baseline.
- `Frozen-backbone mix` is a control-safety diagnostic, not the main offline forecasting method.
| Control-aware fusion | late-frozen anchor + PHF terminal candidate |  |  |  |  |  | Can we keep late-frozen control behavior while recovering PHF terminal gains? |

## Main Reading

- `Horizon mixture` is the current offline PHF representative and CO2 forecasting leader.
- `Late frozen expert` remains the strongest current closed-loop CO2 control baseline.
- `Control-aware fusion` is the new single-candidate follow-up that should be judged by both control-relevant rank and `GradientMPC` transfer, not offline CO2 alone.
- `Recoupled expert` remains the strongest current overall closed-loop objective baseline.
- `Frozen-backbone mix` is a control-safety diagnostic, not the main offline forecasting method.
