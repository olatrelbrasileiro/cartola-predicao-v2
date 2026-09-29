# P1.3 — Auditoria numerica 2025

- Gerado em: 2026-09-21T12:56:47
- Arquivos RAW: 38

## 1. Comparacao de chaves (rodada, atleta_id) RAW x canonico

| rodada | n_raw | n_can | only_raw | only_can |
|---|---|---|---|---|
| 1 | 619 | 619 | 0 | 0 |
| 10 | 720 | 720 | 0 | 0 |
| 11 | 726 | 726 | 0 | 0 |
| 12 | 727 | 727 | 0 | 0 |
| 13 | 718 | 718 | 0 | 0 |
| 14 | 724 | 724 | 0 | 0 |
| 15 | 730 | 730 | 0 | 0 |
| 16 | 732 | 732 | 0 | 0 |
| 17 | 740 | 740 | 0 | 0 |
| 18 | 742 | 742 | 0 | 0 |
| 19 | 747 | 747 | 0 | 0 |
| 2 | 674 | 674 | 0 | 0 |
| 20 | 756 | 756 | 0 | 0 |
| 21 | 767 | 767 | 0 | 0 |
| 22 | 774 | 774 | 0 | 0 |
| 23 | 774 | 774 | 0 | 0 |
| 24 | 775 | 775 | 0 | 0 |
| 25 | 786 | 786 | 0 | 0 |
| 26 | 787 | 787 | 0 | 0 |
| 27 | 789 | 789 | 0 | 0 |
| 28 | 797 | 797 | 0 | 0 |
| 29 | 798 | 798 | 0 | 0 |
| 3 | 687 | 687 | 0 | 0 |
| 30 | 800 | 800 | 0 | 0 |
| 31 | 802 | 802 | 0 | 0 |
| 32 | 801 | 801 | 0 | 0 |
| 33 | 802 | 802 | 0 | 0 |
| 34 | 803 | 803 | 0 | 0 |
| 35 | 805 | 805 | 0 | 0 |
| 36 | 806 | 806 | 0 | 0 |
| 37 | 807 | 807 | 0 | 0 |
| 38 | 819 | 819 | 0 | 0 |
| 4 | 695 | 695 | 0 | 0 |
| 5 | 703 | 703 | 0 | 0 |
| 6 | 712 | 712 | 0 | 0 |
| 7 | 714 | 714 | 0 | 0 |
| 8 | 714 | 714 | 0 | 0 |
| 9 | 717 | 717 | 0 | 0 |

Nenhuma divergencia de chaves.

## 2. Dtypes RAW (inferido) vs canonico

- colunas auditadas: 380
- float_by_nan (esperado, nao e' problema): 0
- perda real de tipo (float sem NaN): 0

## 3. Categorizacao da representacao bruta (por rodada x coluna)

              INTEGER  FLOAT_DECIMAL  SCIENTIFIC  FLOAT_OTHER  EMPTY  NULL_LITERAL  NON_NUMERIC
coluna                                                                                         
atleta_id       28589              0           0            0      0             0            0
clube_id        28589              0           0            0      0             0            0
jogos_num       28589              0           0            0      0             0            0
media_num         619          27970           0            0      0             0            0
pontos_num        619          27970           0            0      0             0            0
posicao_id      28589              0           0            0      0             0            0
preco_num         619          27970           0            0      0             0            0
rodada          28589              0           0            0      0             0            0
status_id       28589              0           0            0      0             0            0
variacao_num      619          27970           0            0      0             0            0

## 4. Duplicatas (rodada, atleta_id) no RAW

Nenhuma.

## 5. Comparacao numerica por coluna

| coluna | tipo | comparadas | diffs | max_abs |
|---|---|---|---|---|
| atleta_id | int | (index) | 0 | 0.00e+00 |
| rodada | int | 28589 | 0 | 0.00e+00 |
| clube_id | int | 28589 | 0 | 0.00e+00 |
| posicao_id | int | 28589 | 0 | 0.00e+00 |
| status_id | int | 28589 | 0 | 0.00e+00 |
| jogos_num | int | 28589 | 0 | 0.00e+00 |
| pontos_num | float | 28589 | 0 | 0.00e+00 |
| media_num | float | 28589 | 0 | 0.00e+00 |
| preco_num | float | 28589 | 0 | 0.00e+00 |
| variacao_num | float | 28589 | 0 | 0.00e+00 |

## 6. Divergencias de contagem por rodada

Nenhuma.

## 7. Top deltas concretos por coluna

Nenhum delta acima de tolerancia.

## 8. Classificacao

**PASS** — sem divergencia estrutural nem numerica.

## Integridade

- Parquet inalterado: True
