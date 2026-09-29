# EXP-12 — Team Strength Ratings V3 (FTT/FS Proxy + Histórico Completo)

## Resultado do poder preditivo

| feature                   |    spearman |    n |   coverage |
|:--------------------------|------------:|-----:|-----------:|
| team_gols_sofridos_mean_5 | -0.0520293  | 1864 |   0.978478 |
| gk_pontos_mean_5          |  0.0491439  | 1769 |   0.928609 |
| team_gols_sofridos_mean_3 | -0.0430521  | 1864 |   0.978478 |
| rating_attack_own         |  0.0400662  | 1864 |   0.978478 |
| rating_defense_opp        |  0.0316141  | 1864 |   0.978478 |
| opp_gols_marcados_mean_5  | -0.016486   | 1864 |   0.978478 |
| soma_forca                |  0.0100971  | 1864 |   0.978478 |
| dif_forca                 |  0.00850186 | 1864 |   0.978478 |
| opp_gols_marcados_mean_3  |  0.00591249 | 1864 |   0.978478 |
| rating_defense_own        | -0.00546829 | 1864 |   0.978478 |
| rating_attack_opp         |  0.00542733 | 1864 |   0.978478 |

## Interpretação
- A cobertura deve ter saltado de ~25% para >90% ao incluir os dados de 2024 e 2025.
- Se o Spearman de `soma_forca` e `rating_attack_opp` se mantiver superior às médias simples 
  com essa amostra muito maior, temos uma feature candidata definitiva para o modelo.
