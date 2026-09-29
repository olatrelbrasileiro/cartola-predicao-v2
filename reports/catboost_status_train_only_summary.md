# CatBoost status-train-only — ablacao (import-based)

- Data: 2026-09-22T16:41:41.521326
- Pipeline base: `scripts/catboost_v2.py` (importado, nao copiado)
- Unica divergencia: `tr = tr[tr['status_id'].isin([7, 2])]`
- status_id recuperado como METADATA (merge 1:1), nao como feature
- Harness: 76 folds

## Retencao do treino

| metrica | valor |
|---|---|
| n_train_original | 504821 |
| n_train_7_2 | 315062 |
| pct_retido medio | 62.55% |

## Performance

| Modelo | MAE | RMSE | Spearman |
|---|---|---|---|
| status-train (COM) | 2.7511 | 3.7221 | 0.2840 |
| status-train (SEM) | 2.7652 | — | 0.2686 |
| B2S P0-corrected | 2.7485 | — | 0.2552 |

