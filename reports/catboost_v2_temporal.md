# C2 — Diagnostico temporal CatBoost v2

- Gerado em: 2026-09-21T22:50:41
- Fonte: `reports/catboost_v2_predictions.csv`
- Observacoes: 23691
- Folds: 76
- SHA Parquet: `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`

## 1. Resultado por temporada

| temporada | n | folds | MAE_CB | MAE_B2S | delta_MAE | RMSE_CB | RMSE_B2S | delta_RMSE | SP_CB | SP_B2S | delta_SP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 9084 | 29 | 2.5425 | 2.6150 | -0.0725 | 3.5868 | 3.6292 | -0.0423 | 0.3143 | 0.2443 | +0.0700 |
| 2025 | 8881 | 29 | 2.7403 | 2.8420 | -0.1017 | 3.8866 | 3.8995 | -0.0129 | 0.3300 | 0.2543 | +0.0757 |
| 2026 | 5726 | 18 | 2.7269 | 2.8153 | -0.0884 | 3.8699 | 3.9300 | -0.0601 | 0.3282 | 0.2607 | +0.0675 |

## 2. Estabilidade por fold

| temporada | folds | CB melhor | CB pior | empates | frac melhor | delta medio | mediana | std | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 29 | 25 | 4 | 0 | 0.862 | -0.0709 | -0.0509 | 0.0817 | -0.2366 | +0.1141 |
| 2025 | 29 | 23 | 6 | 0 | 0.793 | -0.0988 | -0.0889 | 0.1101 | -0.3408 | +0.0957 |
| 2026 | 18 | 16 | 2 | 0 | 0.889 | -0.0922 | -0.0918 | 0.0908 | -0.2407 | +0.1266 |

## 3. NDCG por temporada

| temporada | NDCG@3 CB | B2S | delta | NDCG@5 CB | B2S | delta | NDCG@7 CB | B2S | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.4460 | 0.4289 | 0.0170 | 0.4667 | 0.4523 | 0.0144 | 0.4891 | 0.4692 | 0.0198 |
| 2025 | 0.4729 | 0.4430 | 0.0299 | 0.4877 | 0.4659 | 0.0218 | 0.5076 | 0.4847 | 0.0229 |
| 2026 | 0.4368 | 0.4011 | 0.0358 | 0.4649 | 0.4220 | 0.0429 | 0.4891 | 0.4450 | 0.0441 |

## 4. Heterogeneidade

- CatBoost melhor em MAE: [2024, 2025, 2026]
- CatBoost pior em MAE: []
- Sinal de delta MAE misto: False
- CatBoost melhor em Spearman: [2024, 2025, 2026]
- CatBoost pior em Spearman: []
- Sinal de delta Spearman misto: False

## 5. Alertas NDCG

- total: 0
- n<2: 0
- IDCG=0: 0
- IDCG<0: 0
- NDCG<0: 0
- resultado_ndcg: **PASS**

