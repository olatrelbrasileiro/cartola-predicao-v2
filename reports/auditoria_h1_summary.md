# Auditoria H1 — B3: compressão vs informação

- Inicio: 2026-09-20T08:52:42
- Fim: 2026-09-20T08:52:44
- Duracao: 1.67s
- Observações: 58182
- Folds: 76

## Referências

- B2 MAE: 1.5778  Spearman: 0.4617
- B3 MAE: 1.5339  Spearman: 0.3454

## H1.1 — B2 comprimido em direção a train_median

`B2c(α) = α·B2 + (1−α)·train_median`  (Spearman preservado p/ α>0)

 alpha  mae_global  rmse_global
   0.0    1.485736     3.309168
   0.1    1.450591     3.181452
   0.2    1.423562     3.067741
   0.3    1.406607     2.969644
   0.4    1.400183     2.888752
   0.5    1.404828     2.826544
   0.6    1.420310     2.784271
   0.7    1.446186     2.762849
   0.8    1.481098     2.762762
   0.9    1.525263     2.784014
   1.0    1.577847     2.826122

- Melhor α (menor MAE): 0.4
- MAE em α*: 1.4002

## H1.2 — B3 permutado dentro do fold

 seed  mae_global
   42    1.839869
  123    1.841581
  777    1.843118
 2024    1.832277
 2025    1.849491

- MAE permutado (média): 1.8413
- MAE permutado (min/max): 1.8323 / 1.8495

## H1.3 — B2 com mean/std igualados ao B3

- MAE B2c_std: 1.4589
- Spearman preservado (linear monotônica): 0.4617

## Comparações diretas

- B3 vs B2c(α*): MAE 1.5339 vs 1.4002  delta=+0.1338
- B3 vs B3_perm: MAE 1.5339 vs 1.8413  delta=-0.3073
- B3 vs B2c_std: MAE 1.5339 vs 1.4589  delta=+0.0750

## Imutabilidade

- parquet: OK
- b2_folds: OK
- b3_folds: OK
- erros: OK
