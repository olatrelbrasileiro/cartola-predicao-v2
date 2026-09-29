# Auditoria final da Etapa 5

- Inicio: 2026-09-20T09:04:21
- Fim: 2026-09-20T09:04:21

## Arquivos

| artefato | existe | bytes |
|---|---|---|
| b0 | SIM | 1316 |
| b1 | SIM | 1708 |
| b2 | SIM | 1559 |
| b3 | SIM | 2083 |
| b2s | SIM | 873 |
| b2_spot | SIM | 2693 |
| b3_spot | SIM | 5132 |
| b2s_spec | SIM | 1112 |
| b2s_specmd | SIM | 1090 |
| scouts | SIM | 3186 |
| 3c_md | SIM | 2052 |
| 3c_csv | SIM | 651 |
| h1 | SIM | 1424 |
| h2a | SIM | 912 |
| h2b | SIM | 1568 |
| b2b3 | SIM | 7301 |
| b2b3_erros | SIM | 6899080 |

## Métricas oficiais (recalculadas dos CSVs)

- B0  MAE = 2.0748
- B1  MAE = 1.6012
- B2  MAE = 1.5778
- B3  MAE = 1.5339  (diagnostico)
- B2S MAE = 1.4016  Spearman = 0.4617

## Coerências

- 76 folds em todos os baselines: OK
- α médio H2A = 0.401; α médio B2S = 0.401 (diferença = 0.0000)
- Parquet: linhas=77253 colunas=38  OK=True
- Parquet SHA: 15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4

## Falhas

- nenhuma
