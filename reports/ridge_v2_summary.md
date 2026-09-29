# Ridge v2 — Summary

- Gerado em: 2026-09-21T13:45:00
- Duracao: 490.3s
- Folds: 76
- Obs no test: 23691
- n_ridge_fits (real): 12052

## Metricas globais

- Ridge v2 MAE ponderado: 2.7873
- Ridge v2 RMSE ponderado: 3.6711
- Ridge v2 Spearman medio: 0.3124
- B2S MAE ponderado: 2.7485
- Delta MAE (Ridge - B2S): +0.0388

## Por fold

- Folds Ridge < B2S (tol 1e-4): 20/76
- Folds empate: 1
- Folds Ridge > B2S: 55
- Mediana do delta: +0.0432

## Por temporada

 temporada  n_folds  n_test  mae_ridge  mae_b2s    delta  spearman_ridge
      2024     29.0  9084.0   2.675411 2.615037 0.060374        0.301828
      2025     29.0  8881.0   2.871731 2.841995 0.029736        0.318953
      2026     18.0  5726.0   2.834059 2.815298 0.018761        0.319070

## Criterios de promocao

- C1 (MAE global): False
- C2 (>= 50/76 folds): False (20/76)
- C3 (3 temporadas): False
- C4 (nenhuma piora > 1.5%): False
- C5 (mediana delta < -1e-3): False

**Decisao: KEEP_B2S**

Motivo: criterios centrais nao atendidos

## Distribuicao de alpha_star

alpha_star
0.1       29
1000.0    47

## Integridade

- canonical: OK
- b2s_folds: OK
- harness: OK
- p0_folds: OK
