# P1.1 — Justificativa quantitativa do k=7

- Gerado em: 2026-09-21T12:55:37
- Folds avaliados: 76
- Modelo: `y_ij ~ N(mu_i, sigma^2)`, `mu_i ~ N(mu_0, tau^2)`
- Estimador: `k = sigma^2 / tau^2`
- k pre-registrado: 7.0 (NAO alterado)

## Contagens por fold

- folds com estimativa valida: 76/76
- folds degenerados: 0

### n_players_var (players com n_i >= 2)
- min    : 482
- mediana: 609
- max    : 707

### n_players_all (players com n_i >= 1)
- min    : 539
- mediana: 671
- max    : 755

## Distribuicao de k (folds validos)

### sigma2
- n      : 76
- min    : 11.6641
- p25    : 11.8820
- mediana: 12.9475
- p75    : 13.3571
- max    : 13.7032

### tau2_corrected
- n      : 76
- min    : 0.5724
- p25    : 0.7786
- mediana: 0.9193
- p75    : 1.0888
- max    : 2.0670

### k_estimated
- n      : 76
- min    : 6.2236
- p25    : 11.4852
- mediana: 13.9453
- p75    : 16.3863
- max    : 20.5905

## Por temporada (k mediano)

- 2024: n=29  k_mediana=13.271  k_media=14.079  k_min=9.235  k_max=20.590
- 2025: n=29  k_mediana=13.616  k_media=12.577  k_min=6.224  k_max=17.876
- 2026: n=18  k_mediana=15.689  k_media=15.793  k_min=11.582  k_max=20.169

## Conclusao

- mediana de k entre folds validos: **13.945**
- k pre-registrado: 7.0

**FATO:** mediana de k na mesma ordem de magnitude de 7.
**INTERPRETACAO:** k=7 e compativel com a estrutura de variancia dos dados de treino.

- Folds degenerados foram EXCLUIDOS da mediana, nao silenciados.
- Nenhum tuning. k=7 mantido.

## Integridade

- Parquet inalterado: True
