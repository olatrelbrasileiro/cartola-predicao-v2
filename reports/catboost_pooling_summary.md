# CatBoost pooling cumulativo entre temporadas — ablacao

- Data: 2026-10-08T20:24:22.999904
- Pipeline base: `scripts/catboost_v2.py` (importado)
- Unica divergencia: `tr_all` inclui temporadas < t
- Features: inalteradas (reset por temporada, atleta_id)
- Teste: identico ao v2
- status_id: fora das features, sem filtro
- Harness: 76 folds

## Treino (medias)

| metrica | valor |
|---|---|
| n_train_v2 medio | 6642.4 |
| n_train_pooling medio | 16661.5 |
| n_train_temp_atual medio | 6642.4 |
| n_train_temporadas_anteriores medio | 10019.1 |

## Performance

| Modelo | MAE | RMSE | Spearman |
|---|---|---|---|
| pooling (COM) | 2.6322 | 3.7483 | 0.3416 |
| pooling (SEM) | 2.6556 | 3.7839 | 0.3258 |
| CatBoost v2 (COM) | 2.6629 | — | 0.3264 |
| B2S P0-corrected | 2.7485 | — | 0.2552 |

## Por temporada

| temporada | n_folds | MAE pool | SP pool | n_train_v2 | n_train_pool |
|---|---|---|---|---|---|
| 2024 | 29 | 2.5416 | 0.3179 | 7114 | 7114 |
| 2025 | 29 | 2.7008 | 0.3533 | 6859 | 18657 |
| 2026 | 18 | 2.6676 | 0.3609 | 5534 | 28829 |

## Comparacao por fold vs v2 (COM)

- Folds: 76
- ΔMAE medio:   -0.0307
- ΔMAE mediano: -0.0000
- ΔMAE min:     -0.2138
- ΔMAE max:     +0.0648
- ΔMAE std:     0.0534

- Exp melhor MAE: 38/76
- Exp pior MAE:   11/76
- Empates MAE:    27/76
