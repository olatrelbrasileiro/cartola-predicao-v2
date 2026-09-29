# EXP-12 — Team Strength Ratings (Poisson/Maher)

## Resultado do poder preditivo

| feature                   |     spearman |    n |   coverage |
|:--------------------------|-------------:|-----:|-----------:|
| rating_attack_own         |  0.100983    |  459 |   0.240945 |
| rating_defense_own        |  0.0854437   |  459 |   0.240945 |
| dif_forca                 | -0.0563141   |  459 |   0.240945 |
| rating_defense_opp        | -0.0535255   |  459 |   0.240945 |
| team_gols_sofridos_mean_5 | -0.0520293   | 1864 |   0.978478 |
| soma_forca                |  0.0518402   |  459 |   0.240945 |
| gk_pontos_mean_5          |  0.0491439   | 1769 |   0.928609 |
| team_gols_sofridos_mean_3 | -0.0430521   | 1864 |   0.978478 |
| opp_gols_marcados_mean_5  | -0.016486    | 1864 |   0.978478 |
| opp_gols_marcados_mean_3  |  0.00591249  | 1864 |   0.978478 |
| rating_attack_opp         |  0.000471775 |  459 |   0.240945 |

## Interpretação

- Se `dif_forca` ou `rating_attack_opp` tiverem Spearman claramente melhor que as médias simples atuais,
  os ratings estão capturando sinal útil.
- Valores absolutos ainda devem ser baixos (problema difícil), mas a comparação relativa importa.
