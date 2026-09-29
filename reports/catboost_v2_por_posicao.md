# Diagnostico CatBoost v2 por posicao

- Gerado em: 2026-09-21T22:28:37
- Fonte: `reports/catboost_v2_predictions.csv` (23691 obs)
- Folds: 76
- SHA Parquet: `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`

## 1. Metodologia

- MAE / Spearman por posicao (todos os folds agregados).
- NDCG@K, K em {3, 5, 7}, grupo = `(temporada, rodada, posicao_id)`.
- Relevancia = `pontos_num` bruto, ganho linear (`DCG@K = sum rel_i / log2(i+1)`).
- Status por grupo:
  - `ok`: NDCG >= 0, incluido na media.
  - `ndcg_negativo`: NDCG < 0, INCLUIDO na media, flagrado.
  - `idcg_zero`: IDCG == 0, excluido.
  - `idcg_negativo`: IDCG < 0, excluido.
  - `n_lt_2`: n < 2, excluido.
- Media simples entre grupos validos.

## 2. MAE por posicao

| posicao | n | folds | MAE_CB | MAE_B2S | delta |
|---|---|---|---|---|---|
| Goleiro | 1433 | 76 | 3.2868 | 3.2513 | +0.0355 |
| Lateral | 3607 | 76 | 3.0224 | 3.0663 | -0.0439 |
| Zagueiro | 3363 | 76 | 2.6720 | 2.7169 | -0.0449 |
| Meia | 7685 | 76 | 2.3340 | 2.4085 | -0.0745 |
| Atacante | 6215 | 76 | 2.8823 | 2.9794 | -0.0971 |
| Tecnico | 1388 | 76 | 1.8719 | 2.3288 | -0.4570 |

## 3. Spearman por posicao

| posicao | Spearman_CB | Spearman_B2S | delta |
|---|---|---|---|
| Goleiro | 0.0849 | 0.0681 | +0.0168 |
| Lateral | 0.2020 | 0.1483 | +0.0537 |
| Zagueiro | 0.2016 | 0.0964 | +0.1052 |
| Meia | 0.2807 | 0.2312 | +0.0496 |
| Atacante | 0.3136 | 0.2593 | +0.0543 |
| Tecnico | 0.2851 | 0.1347 | +0.1504 |

## 4. NDCG@K por posicao

| posicao | NDCG@3 CB | NDCG@3 B2S | NDCG@5 CB | NDCG@5 B2S | NDCG@7 CB | NDCG@7 B2S |
|---|---|---|---|---|---|---|
| Goleiro | 0.4274 | 0.4138 | 0.4564 | 0.4571 | 0.4931 | 0.4861 |
| Lateral | 0.4089 | 0.3962 | 0.4319 | 0.4293 | 0.4517 | 0.4440 |
| Zagueiro | 0.3838 | 0.3466 | 0.4136 | 0.3670 | 0.4378 | 0.3856 |
| Meia | 0.4076 | 0.3692 | 0.4121 | 0.3806 | 0.4216 | 0.3918 |
| Atacante | 0.4124 | 0.3892 | 0.4212 | 0.4034 | 0.4335 | 0.4142 |
| Tecnico | 0.6844 | 0.6511 | 0.7107 | 0.6645 | 0.7391 | 0.6947 |

### Grupos validos / invalidos por posicao

| posicao | K | validos_cb | invalidos_cb | validos_b2s | invalidos_b2s |
|---|---|---|---|---|---|
| Goleiro | 3 | 76 | 0 | 76 | 0 |
| Goleiro | 5 | 76 | 0 | 76 | 0 |
| Goleiro | 7 | 76 | 0 | 76 | 0 |
| Lateral | 3 | 76 | 0 | 76 | 0 |
| Lateral | 5 | 76 | 0 | 76 | 0 |
| Lateral | 7 | 76 | 0 | 76 | 0 |
| Zagueiro | 3 | 76 | 0 | 76 | 0 |
| Zagueiro | 5 | 76 | 0 | 76 | 0 |
| Zagueiro | 7 | 76 | 0 | 76 | 0 |
| Meia | 3 | 76 | 0 | 76 | 0 |
| Meia | 5 | 76 | 0 | 76 | 0 |
| Meia | 7 | 76 | 0 | 76 | 0 |
| Atacante | 3 | 76 | 0 | 76 | 0 |
| Atacante | 5 | 76 | 0 | 76 | 0 |
| Atacante | 7 | 76 | 0 | 76 | 0 |
| Tecnico | 3 | 76 | 0 | 76 | 0 |
| Tecnico | 5 | 76 | 0 | 76 | 0 |
| Tecnico | 7 | 76 | 0 | 76 | 0 |

## 5. Alertas de NDCG

- Total de grupos avaliados (todas as posicoes): 456
- Alertas totais (status != 'ok'): 0
  - `n_lt_2`: 0
  - `idcg_zero`: 0
  - `idcg_negativo`: 0
  - `ndcg_negativo`: 0

Nenhum alerta. NDCG uniformemente valido.

## 6. Estabilidade por temporada

| temporada | posicao | n | MAE_CB | MAE_B2S | delta_MAE | Spearman_CB | Spearman_B2S | delta_SP |
|---|---|---|---|---|---|---|---|---|
| 2024 | Goleiro | 550 | 3.1227 | 3.0718 | +0.0510 | 0.0719 | 0.0438 | +0.0281 |
| 2024 | Lateral | 1375 | 2.8909 | 2.8945 | -0.0036 | 0.1518 | 0.1004 | +0.0514 |
| 2024 | Zagueiro | 1302 | 2.5976 | 2.6253 | -0.0277 | 0.1362 | 0.0447 | +0.0914 |
| 2024 | Meia | 3044 | 2.2212 | 2.2824 | -0.0611 | 0.2938 | 0.2453 | +0.0485 |
| 2024 | Atacante | 2288 | 2.7643 | 2.8560 | -0.0917 | 0.3284 | 0.2802 | +0.0482 |
| 2024 | Tecnico | 525 | 1.7814 | 2.2578 | -0.4764 | 0.3013 | 0.1130 | +0.1882 |
| 2025 | Goleiro | 537 | 3.4477 | 3.4084 | +0.0393 | 0.0886 | 0.0795 | +0.0091 |
| 2025 | Lateral | 1372 | 3.1270 | 3.2095 | -0.0826 | 0.2177 | 0.1646 | +0.0531 |
| 2025 | Zagueiro | 1241 | 2.7520 | 2.8116 | -0.0596 | 0.2050 | 0.0935 | +0.1115 |
| 2025 | Meia | 2804 | 2.3379 | 2.4510 | -0.1131 | 0.2916 | 0.2205 | +0.0711 |
| 2025 | Atacante | 2402 | 2.9826 | 3.0728 | -0.0902 | 0.2557 | 0.2197 | +0.0360 |
| 2025 | Tecnico | 525 | 2.0188 | 2.4061 | -0.3873 | 0.2757 | 0.0682 | +0.2075 |
| 2026 | Goleiro | 346 | 3.2981 | 3.2928 | +0.0052 | 0.0559 | 0.0180 | +0.0380 |
| 2026 | Lateral | 860 | 3.0659 | 3.1127 | -0.0467 | 0.2222 | 0.1435 | +0.0787 |
| 2026 | Zagueiro | 820 | 2.6691 | 2.7191 | -0.0500 | 0.2116 | 0.0595 | +0.1520 |
| 2026 | Meia | 1837 | 2.5150 | 2.5526 | -0.0376 | 0.2425 | 0.2404 | +0.0022 |
| 2026 | Atacante | 1525 | 2.9012 | 3.0173 | -0.1162 | 0.3707 | 0.2925 | +0.0782 |
| 2026 | Tecnico | 338 | 1.7841 | 2.3191 | -0.5350 | 0.2432 | 0.1648 | +0.0784 |

## 7. Estabilidade por fold

| posicao | n_folds | CB melhor | CB pior | empate | delta_mediana | delta_std |
|---|---|---|---|---|---|---|
| Goleiro | 76 | 37 | 39 | 0 | +0.0158 | 0.2714 |
| Lateral | 76 | 45 | 31 | 0 | -0.0494 | 0.2007 |
| Zagueiro | 76 | 46 | 30 | 0 | -0.0729 | 0.1919 |
| Meia | 76 | 58 | 18 | 0 | -0.0679 | 0.1075 |
| Atacante | 76 | 56 | 19 | 1 | -0.0862 | 0.1248 |
| Tecnico | 76 | 70 | 6 | 0 | -0.4790 | 0.2929 |

## 8. Limitacoes

- Diagnostico NAO promove nem despromove CatBoost v2.
- Posicoes com pouca amostra: interpretar com cautela.

## 9. Validacao

- parquet_sha_ok: OK
- folds_76_ok: OK
- sem_duplicatas: OK
- participantes_only: OK
- predicoes_validas: OK
- posicoes_validas: OK
- mae_por_posicao_ok: OK
- spearman_por_posicao_ok: OK
- ndcg_por_posicao_ok: OK
- comparacao_b2s_mesmo_conjunto_ok: OK
- relatorio_gerado: OK
- validation_json_gerado: OK

**Resultado metricas (MAE/Spearman): PASS**
**Resultado NDCG: PASS**
