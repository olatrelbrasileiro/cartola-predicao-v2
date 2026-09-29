# EXP-12 — Team Strength Ratings V2 (FTT/FS Proxy)

## Resultado do poder preditivo

| feature                   |    spearman |    n |   coverage |
|:--------------------------|------------:|-----:|-----------:|
| soma_forca                |  0.0744895  |  477 |   0.250394 |
| rating_attack_opp         |  0.0691679  |  477 |   0.250394 |
| rating_defense_opp        |  0.0587806  |  477 |   0.250394 |
| team_gols_sofridos_mean_5 | -0.0520293  | 1864 |   0.978478 |
| gk_pontos_mean_5          |  0.0491439  | 1769 |   0.928609 |
| team_gols_sofridos_mean_3 | -0.0430521  | 1864 |   0.978478 |
| dif_forca                 |  0.0358019  |  477 |   0.250394 |
| rating_defense_own        |  0.0239183  |  477 |   0.250394 |
| opp_gols_marcados_mean_5  | -0.016486   | 1864 |   0.978478 |
| rating_attack_own         | -0.0107283  |  477 |   0.250394 |
| opp_gols_marcados_mean_3  |  0.00591249 | 1864 |   0.978478 |

## Interpretação
- A mudança de 'Gols' para 'Finalizações (FTT/FS)' como alvo do modelo Poisson 
  deve ter aumentado drasticamente a cobertura (de 24% para >80%).
- Se o Spearman de `rating_defense_own` e `dif_forca` se mantiver superior às 
  médias simples, temos um candidato forte a feature definitiva.
