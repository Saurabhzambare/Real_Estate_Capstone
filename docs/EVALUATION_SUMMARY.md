# Modern regression evaluation

The modern workflow estimates mean monthly mortgage and owner cost for a geographic observation. Its preprocessing is fitted within the training folds. The original capstone remains a separate historical workflow.

| Check | Recorded result |
|---|---|
| Training-only five-fold comparison | Random Forest mean RMSE 253.67 versus 622.45 for the mean baseline. |
| Primary holdout | 11,432 rows, RMSE 252.27, MAE 177.56, and R² 0.8419. |
| Exact-overlap sensitivity | The same predictions evaluated after excluding 31 overlapping eligible rows gave RMSE 251.48 and R² 0.8425 on 11,401 rows. |

The sensitivity subset is part of the existing holdout. The recorded results do not establish performance on a future period or unseen geography.

See [comparison results](../reports/model_comparison.csv), [overlap sensitivity](../reports/holdout_sensitivity.csv), [state diagnostics](../reports/state_metrics.csv), and the [validated environment](validated_environment.md).
