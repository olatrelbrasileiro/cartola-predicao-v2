# Etapa B — Estudo isolado de k (shrinkage historico)

- Gerado em: 2026-09-21T13:10:09
- Duracao: 3.4s
- Folds: 76
- Grade: [0, 3, 5, 7, 10, 14, 20, 30]
- Referencia: k=7
- Preditor: hist_shrunk(k) puro, fallback B0 no cold-start
- Teste: somente participantes

## Tabela final

| k | MAE pond | Δ MAE vs k=7 | Spearman | folds melhor | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| 0 | 2.8587 | +0.0536 | 0.2808 | 13/76 | 2.7247 | 2.9589 | 2.9158 |
| 3 | 2.8094 | +0.0043 | 0.2906 | 27/76 | 2.6806 | 2.9070 | 2.8625 |
| 5 | 2.8056 | +0.0005 | 0.2935 | 28/76 | 2.6775 | 2.9025 | 2.8586 |
| **7** | **2.8051** | 0 | 0.2950 | — | 2.6774 | 2.9016 | 2.8582 |
| 10 | 2.8066 | +0.0014 | 0.2961 | 38/76 | 2.6791 | 2.9026 | 2.8599 |
| 14 | 2.8098 | +0.0047 | 0.2952 | 32/76 | 2.6828 | 2.9052 | 2.8635 |
| 20 | 2.8151 | +0.0100 | 0.2921 | 31/76 | 2.6888 | 2.9095 | 2.8690 |
| 30 | 2.8229 | +0.0178 | 0.2854 | 26/76 | 2.6973 | 2.9161 | 2.8777 |

## Aplicacao dos 5 criterios

| k | crit1 MAE | crit2 folds | crit3 temporadas | crit4 piora | crit5 mediana | elegivel |
|---|---|---|---|---|---|---|
| 0 | False | False (13/76) | False | False | False (+0.0561) | False |
| 3 | False | False (27/76) | False | True | False (+0.0074) | False |
| 5 | False | False (28/76) | False | True | False (+0.0018) | False |
| 7 | (referencia) | — | — | — | — | — |
| 10 | False | False (38/76) | False | True | True (-0.0001) | False |
| 14 | False | False (32/76) | False | True | False (+0.0028) | False |
| 20 | False | False (31/76) | False | True | False (+0.0070) | False |
| 30 | False | False (26/76) | False | True | False (+0.0145) | False |

## Decisao

- **KEEP_K7**
- Motivo: nenhum candidato cumpriu todos os 5 criterios

- Recomendacao documental. Nenhum script de producao foi alterado.

## Notas descritivas

- k=0 e' o controle sem shrinkage (`hist_shrunk(0) = hist_mean`).
- k=30 esta fora da faixa Empirical-Bayes de P1.1 (6.22–20.59) e foi incluido deliberadamente como teste de extrapolacao/saturacao.
- O efeito de k tende a ser maior para `n` pequeno e desaparecer progressivamente conforme `n` cresce. Propriedade matematica do shrinkage, nao defeito do experimento.
## Integridade

- canonical: OK
- b2s_folds: OK
- harness: OK
- p0_folds: OK
