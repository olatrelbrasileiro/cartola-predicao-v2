# EXP-12 - F2 - Auditoria de Team Context Features

**SHA256 parquet**: `26d837ba62ea2d13146abd88f660893ff85f10c351a1bed80a66e3ef3116b6d2`

## 1. Estrutura de entrada

- arquivo: `data/interim/exp12/team_match_history.parquet`
- shape: (514, 27)
- colunas: ['temporada', 'rodada', 'partida_id', 'data', 'clube_id', 'clube_nome', 'adversario_id', 'adversario_nome', 'mando', 'gols_marcados', 'gols_sofridos', 'DP', 'GS', 'DS', 'SG', 'FC', 'G', 'A', 'FTT', 'FA', 'FS', 'I', 'PS', 'FD', 'FF', 'FT', 'PP']
- temporadas: [2026]
- rodadas unicas: 26 min=1 max=26
- clubes: 20
- duplicatas (T,R,clube): 0

## 2. Saida

- shape: (514, 24)
- colunas: ['temporada', 'rodada', 'clube_id', 'opp_clube_id', 'team_goals_scored_last3', 'team_goals_scored_last5', 'team_goals_conceded_last3', 'team_goals_conceded_last5', 'team_clean_sheet_rate_last3', 'team_clean_sheet_rate_last5', 'team_shots_last3', 'team_shots_last5', 'team_shots_on_target_last3', 'team_shots_on_target_last5', 'opp_goals_scored_last3', 'opp_goals_scored_last5', 'opp_goals_conceded_last3', 'opp_goals_conceded_last5', 'opp_clean_sheet_rate_last3', 'opp_clean_sheet_rate_last5', 'opp_shots_last3', 'opp_shots_last5', 'opp_shots_on_target_last3', 'opp_shots_on_target_last5']
- duplicatas (T,R,clube): 0

## 3. NaN por feature

- `team_goals_scored_last3`: 20
- `team_goals_scored_last5`: 20
- `team_goals_conceded_last3`: 20
- `team_goals_conceded_last5`: 20
- `team_clean_sheet_rate_last3`: 20
- `team_clean_sheet_rate_last5`: 20
- `team_shots_last3`: 20
- `team_shots_last5`: 20
- `team_shots_on_target_last3`: 20
- `team_shots_on_target_last5`: 20
- `opp_goals_scored_last3`: 20
- `opp_goals_scored_last5`: 20
- `opp_goals_conceded_last3`: 20
- `opp_goals_conceded_last5`: 20
- `opp_clean_sheet_rate_last3`: 20
- `opp_clean_sheet_rate_last5`: 20
- `opp_shots_last3`: 20
- `opp_shots_last5`: 20
- `opp_shots_on_target_last3`: 20
- `opp_shots_on_target_last5`: 20

## 4. Exemplos manuais auditados

### Rodada inicial da temporada
**Alvo**: T=2026 R=1 clube_id=262 opp=276.0

| temporada | rodada | gm | gs | FTT | FA | cs |
|---|---|---|---|---|---|---|

Features finais:

- `team_goals_scored_last3` = nan
- `team_goals_scored_last5` = nan
- `team_goals_conceded_last3` = nan
- `team_goals_conceded_last5` = nan
- `team_clean_sheet_rate_last3` = nan
- `team_clean_sheet_rate_last5` = nan
- `team_shots_last3` = nan
- `team_shots_last5` = nan
- `team_shots_on_target_last3` = nan
- `team_shots_on_target_last5` = nan
- `opp_goals_scored_last3` = nan
- `opp_goals_scored_last5` = nan
- `opp_goals_conceded_last3` = nan
- `opp_goals_conceded_last5` = nan
- `opp_clean_sheet_rate_last3` = nan
- `opp_clean_sheet_rate_last5` = nan
- `opp_shots_last3` = nan
- `opp_shots_last5` = nan
- `opp_shots_on_target_last3` = nan
- `opp_shots_on_target_last5` = nan
### Rodada intermediaria
**Alvo**: T=2026 R=13 clube_id=262 opp=282.0

| temporada | rodada | gm | gs | FTT | FA | cs |
|---|---|---|---|---|---|---|
| 2026 | 8 | 1 | 1 | 8.0 | 1.0 | 0 |
| 2026 | 9 | 0 | 3 | 4.0 | 1.0 | 0 |
| 2026 | 10 | 3 | 1 | 12.0 | 4.0 | 0 |
| 2026 | 11 | 2 | 1 | 12.0 | 8.0 | 0 |
| 2026 | 12 | 2 | 0 | 15.0 | 8.0 | 1 |

Features finais:

- `team_goals_scored_last3` = 2.3333333333333335
- `team_goals_scored_last5` = 1.6
- `team_goals_conceded_last3` = 0.6666666666666666
- `team_goals_conceded_last5` = 1.2
- `team_clean_sheet_rate_last3` = 0.3333333333333333
- `team_clean_sheet_rate_last5` = 0.2
- `team_shots_last3` = 13.0
- `team_shots_last5` = 10.2
- `team_shots_on_target_last3` = 6.666666666666667
- `team_shots_on_target_last5` = 4.4
- `opp_goals_scored_last3` = 0.6666666666666666
- `opp_goals_scored_last5` = 1.2
- `opp_goals_conceded_last3` = 1.3333333333333333
- `opp_goals_conceded_last5` = 1.0
- `opp_clean_sheet_rate_last3` = 0.0
- `opp_clean_sheet_rate_last5` = 0.2
- `opp_shots_last3` = 11.0
- `opp_shots_last5` = 10.8
- `opp_shots_on_target_last3` = 4.666666666666667
- `opp_shots_on_target_last5` = 4.8
### Partida com missing roster
**Alvo**: T=2026 R=2 clube_id=293 opp=264.0

| temporada | rodada | gm | gs | FTT | FA | cs |
|---|---|---|---|---|---|---|
| 2026 | 1 | 1 | 0 | 6.0 | 2.0 | 1 |

Features finais:

- `team_goals_scored_last3` = 1.0
- `team_goals_scored_last5` = 1.0
- `team_goals_conceded_last3` = 0.0
- `team_goals_conceded_last5` = 0.0
- `team_clean_sheet_rate_last3` = 1.0
- `team_clean_sheet_rate_last5` = 1.0
- `team_shots_last3` = 6.0
- `team_shots_last5` = 6.0
- `team_shots_on_target_last3` = 2.0
- `team_shots_on_target_last5` = 2.0
- `opp_goals_scored_last3` = 1.0
- `opp_goals_scored_last5` = 1.0
- `opp_goals_conceded_last3` = 2.0
- `opp_goals_conceded_last5` = 2.0
- `opp_clean_sheet_rate_last3` = 0.0
- `opp_clean_sheet_rate_last5` = 0.0
- `opp_shots_last3` = 12.0
- `opp_shots_last5` = 12.0
- `opp_shots_on_target_last3` = 5.0
- `opp_shots_on_target_last5` = 5.0
### last3 != last5
**Alvo**: T=2026 R=5 clube_id=262 opp=283.0

| temporada | rodada | gm | gs | FTT | FA | cs |
|---|---|---|---|---|---|---|
| 2026 | 1 | 1 | 2 | 11.0 | 5.0 | 0 |
| 2026 | 2 | 1 | 1 | 9.0 | 4.0 | 0 |
| 2026 | 3 | 2 | 1 | nan | nan | 0 |
| 2026 | 4 | 2 | 0 | nan | nan | 1 |

Features finais:

- `team_goals_scored_last3` = 1.6666666666666667
- `team_goals_scored_last5` = 1.5
- `team_goals_conceded_last3` = 0.6666666666666666
- `team_goals_conceded_last5` = 1.0
- `team_clean_sheet_rate_last3` = 0.3333333333333333
- `team_clean_sheet_rate_last5` = 0.25
- `team_shots_last3` = 9.0
- `team_shots_last5` = 10.0
- `team_shots_on_target_last3` = 4.0
- `team_shots_on_target_last5` = 4.5
- `opp_goals_scored_last3` = 1.3333333333333333
- `opp_goals_scored_last5` = 1.0
- `opp_goals_conceded_last3` = 1.6666666666666667
- `opp_goals_conceded_last5` = 2.25
- `opp_clean_sheet_rate_last3` = 0.0
- `opp_clean_sheet_rate_last5` = 0.0
- `opp_shots_last3` = 9.666666666666666
- `opp_shots_last5` = 10.0
- `opp_shots_on_target_last3` = 3.6666666666666665
- `opp_shots_on_target_last5` = 3.75
### Adversario com historico
**Alvo**: T=2026 R=2 clube_id=262 opp=285.0

| temporada | rodada | gm | gs | FTT | FA | cs |
|---|---|---|---|---|---|---|
| 2026 | 1 | 1 | 2 | 11.0 | 5.0 | 0 |

Features finais:

- `team_goals_scored_last3` = 1.0
- `team_goals_scored_last5` = 1.0
- `team_goals_conceded_last3` = 2.0
- `team_goals_conceded_last5` = 2.0
- `team_clean_sheet_rate_last3` = 0.0
- `team_clean_sheet_rate_last5` = 0.0
- `team_shots_last3` = 11.0
- `team_shots_last5` = 11.0
- `team_shots_on_target_last3` = 5.0
- `team_shots_on_target_last5` = 5.0
- `opp_goals_scored_last3` = 0.0
- `opp_goals_scored_last5` = 0.0
- `opp_goals_conceded_last3` = 1.0
- `opp_goals_conceded_last5` = 1.0
- `opp_clean_sheet_rate_last3` = 0.0
- `opp_clean_sheet_rate_last5` = 0.0
- `opp_shots_last3` = 13.0
- `opp_shots_last5` = 13.0
- `opp_shots_on_target_last3` = 5.0
- `opp_shots_on_target_last5` = 5.0

## 5. Notas de implementacao

- Eixo temporal = (temporada, rodada). Data fisica ignorada.
- last3/last5 = ultimas 3/5 PARTIDAS anteriores (posicional).
- clean_sheet = (gols_sofridos == 0). Nao usa SG.
- FTT = FD+FF+FT+G+PP ; FA = FD+G. Roster ausente -> NaN.
- CS rate = CS observados / partidas observadas. NaN se 0 obs.
- Mask de ausencia de roster aplicada no layer de features, sem alteracao do F1.
- Fonte do mask: `reports/exp12/f2_missing_roster_raw_audit.csv` (GATE 2, n_jogadores_raw == 0).
