# Ridge v1 — Especificação

Ver `config/ridge_v1_spec.yaml`.

Features (13): hist_mean, hist_std, hist_n, rec_mean_5, rec_mean_3,
rec_std_5, momentum, basic_mean, basic_rec_5, ga_per_game, ga_rec_5,
entrou_rate, is_cold_start.

Correções aplicadas:
- `ga_rec_5` usa min(5, n_obs) como denominador.
- `ga_per_game` usa a última obs elegível antes do alvo (sem leakage).

Benchmark: B2S (MAE 1.4016, Spearman 0.4617).
