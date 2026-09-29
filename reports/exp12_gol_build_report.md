# EXP-12 GOL — build report

- Shape final: (7865, 94)
- GOL total: 7865
- Temporadas em que scouts foram convertidos (acumulado -> delta): [2024, 2025]

## Diagnóstico heurístico de scouts (NÃO é prova)

| scout    |   n_pairs |   pct_decreasing | heuristica              | forca   |
|:---------|----------:|-----------------:|:------------------------|:--------|
| scout_A  |       136 |                0 | provavelmente_acumulado | forte   |
| scout_CA |      1981 |                0 | provavelmente_acumulado | forte   |
| scout_CV |       254 |                0 | provavelmente_acumulado | forte   |
| scout_DE |      3246 |                0 | provavelmente_acumulado | forte   |
| scout_DS |       325 |                0 | provavelmente_acumulado | forte   |
| scout_FC |       227 |                0 | provavelmente_acumulado | forte   |
| scout_FD |        28 |                0 | provavelmente_acumulado | fraca   |
| scout_FF |         0 |              nan | sem_dados               | n/a     |
| scout_FS |      2073 |                0 | provavelmente_acumulado | forte   |
| scout_FT |         0 |              nan | sem_dados               | n/a     |
| scout_G  |        27 |                0 | provavelmente_acumulado | fraca   |
| scout_I  |         0 |              nan | sem_dados               | n/a     |
| scout_PC |       285 |                0 | provavelmente_acumulado | forte   |
| scout_PS |         0 |              nan | sem_dados               | n/a     |
| scout_PP |         0 |              nan | sem_dados               | n/a     |
| scout_SG |      2357 |                0 | provavelmente_acumulado | forte   |
| scout_V  |         0 |              nan | sem_dados               | n/a     |
| scout_DP |       493 |                0 | provavelmente_acumulado | forte   |
| scout_GC |        26 |                0 | provavelmente_acumulado | fraca   |

## Detecção por temporada (auto)

| scout | 2024 | 2025 | 2026 |
|---|---|---|---|
| scout_DE | accumulated | accumulated | accumulated |

## Por temporada

| temporada | linhas | elegíveis | participou=0 | participou>0 | não participou |
|---|---|---|---|---|---|
| 2024 | 2882 | 714 | 75 | 558 | 2168 |
| 2025 | 2976 | 694 | 2 | 604 | 2282 |
| 2026 | 2007 | 497 | 1 | 430 | 1510 |

## Missing por feature

| feature | missing | missing_rate |
|---|---|---|
| team_goals_scored_last3 | 5924 | 0.7532 |
| team_goals_scored_last5 | 5924 | 0.7532 |
| team_goals_conceded_last3 | 5924 | 0.7532 |
| team_goals_conceded_last5 | 5924 | 0.7532 |
| team_clean_sheet_rate_last3 | 5924 | 0.7532 |
| team_clean_sheet_rate_last5 | 5924 | 0.7532 |
| team_shots_last3 | 5924 | 0.7532 |
| team_shots_last5 | 5924 | 0.7532 |
| team_shots_on_target_last3 | 5924 | 0.7532 |
| team_shots_on_target_last5 | 5924 | 0.7532 |
| opp_goals_scored_last3 | 5924 | 0.7532 |
| opp_goals_scored_last5 | 5924 | 0.7532 |
| opp_goals_conceded_last3 | 5924 | 0.7532 |
| opp_goals_conceded_last5 | 5924 | 0.7532 |
| opp_clean_sheet_rate_last3 | 5924 | 0.7532 |
| opp_clean_sheet_rate_last5 | 5924 | 0.7532 |
| opp_shots_last3 | 5924 | 0.7532 |
| opp_shots_last5 | 5924 | 0.7532 |
| opp_shots_on_target_last3 | 5924 | 0.7532 |
| opp_shots_on_target_last5 | 5924 | 0.7532 |
| gk_pontos_last1 | 6096 | 0.7751 |
| gk_pontos_mean_3 | 6096 | 0.7751 |
| gk_pontos_mean_3_n_obs | 5960 | 0.7578 |
| gk_pontos_mean_5 | 6096 | 0.7751 |
| gk_pontos_mean_5_n_obs | 5960 | 0.7578 |
| gk_pontos_mean_hist | 6040 | 0.7680 |
| gk_DE_mean_3 | 6102 | 0.7758 |
| gk_DE_mean_3_n_obs | 5960 | 0.7578 |
| gk_DE_mean_5 | 6102 | 0.7758 |
| gk_DE_mean_5_n_obs | 5960 | 0.7578 |
| team_gols_sofridos_mean_3 | 190 | 0.0242 |
| team_gols_sofridos_mean_3_n_obs | 0 | 0.0000 |
| team_gols_sofridos_mean_5 | 190 | 0.0242 |
| team_gols_sofridos_mean_5_n_obs | 0 | 0.0000 |
| team_clean_sheet_rate_mean_3 | 190 | 0.0242 |
| team_clean_sheet_rate_mean_3_n_obs | 0 | 0.0000 |
| team_clean_sheet_rate_mean_5 | 190 | 0.0242 |
| team_clean_sheet_rate_mean_5_n_obs | 0 | 0.0000 |
| opp_gols_marcados_mean_3 | 190 | 0.0242 |
| opp_gols_marcados_mean_3_n_obs | 0 | 0.0000 |
| opp_gols_marcados_mean_5 | 190 | 0.0242 |
| opp_gols_marcados_mean_5_n_obs | 0 | 0.0000 |
| opp_FTT_mean_3 | 301 | 0.0383 |
| opp_FTT_mean_3_n_obs | 0 | 0.0000 |
| opp_FTT_mean_5 | 287 | 0.0365 |
| opp_FTT_mean_5_n_obs | 0 | 0.0000 |
| odd_p_home | 0 | 0.0000 |
| odd_p_draw | 0 | 0.0000 |
| odd_p_away | 0 | 0.0000 |

## Cobertura do master

- chaves GOL distintas: 2060
- chaves ausentes: 26

## Auditoria manual


### 2024 R19 — atleta_id=37656 clube=266 adv=275 mando=C
  gk_pontos_last1                                = 10.0000
  gk_pontos_mean_3                               = 5.3333
  gk_pontos_mean_3_n_obs                         = 3.0000
  gk_pontos_mean_5                               = 5.0000
  gk_pontos_mean_5_n_obs                         = 5.0000
  gk_pontos_mean_hist                            = 3.6562
  gk_DE_mean_3                                   = 4.3333
  gk_DE_mean_3_n_obs                             = 3.0000
  gk_DE_mean_5                                   = 4.8000
  gk_DE_mean_5_n_obs                             = 5.0000
  team_gols_sofridos_mean_3                      = 0.3333
  team_gols_sofridos_mean_3_n_obs                = 3
  team_gols_sofridos_mean_5                      = 0.6000
  team_gols_sofridos_mean_5_n_obs                = 5
  team_clean_sheet_rate_mean_3                   = 0.6667
  team_clean_sheet_rate_mean_3_n_obs             = 3
  team_clean_sheet_rate_mean_5                   = 0.4000
  team_clean_sheet_rate_mean_5_n_obs             = 5
  opp_gols_marcados_mean_3                       = 1.6667
  opp_gols_marcados_mean_3_n_obs                 = 3
  opp_gols_marcados_mean_5                       = 1.8000
  opp_gols_marcados_mean_5_n_obs                 = 5
  opp_FTT_mean_3                                 = 13.0000
  opp_FTT_mean_3_n_obs                           = 3
  opp_FTT_mean_5                                 = 13.0000
  opp_FTT_mean_5_n_obs                           = 5
  --- team window k=3: rounds R16..R18 (encontradas=3, n_obs=3)
      R16  gols_sofridos=1  clean_sheet=0
      R17  gols_sofridos=0  clean_sheet=1
      R18  gols_sofridos=0  clean_sheet=1
  --- team window k=5: rounds R14..R18 (encontradas=5, n_obs=5)
      R14  gols_sofridos=1  clean_sheet=0
      R15  gols_sofridos=1  clean_sheet=0
      R16  gols_sofridos=1  clean_sheet=0
      R17  gols_sofridos=0  clean_sheet=1
      R18  gols_sofridos=0  clean_sheet=1
  --- opp window k=3: rounds R16..R18 (encontradas=3)
      R16  gols_marcados=3  FTT=16
      R17  gols_marcados=0  FTT=11
      R18  gols_marcados=2  FTT=12
  --- opp window k=5: rounds R14..R18 (encontradas=5)
      R14  gols_marcados=2  FTT=15
      R15  gols_marcados=2  FTT=11
      R16  gols_marcados=3  FTT=16
      R17  gols_marcados=0  FTT=11
      R18  gols_marcados=2  FTT=12
  --- gk window k=3: rounds R16..R18 (participações encontradas=2)
      R16  pontos_num=2.0  scout_DE=70.0000  
      R18  pontos_num=10.0  scout_DE=75.0000  
  --- gk window k=5: rounds R14..R18 (participações encontradas=4)
      R14  pontos_num=2.0  scout_DE=62.0000  
      R15  pontos_num=4.0  scout_DE=67.0000  
      R16  pontos_num=2.0  scout_DE=70.0000  
      R18  pontos_num=10.0  scout_DE=75.0000  

### 2024 R19 — atleta_id=68928 clube=276 adv=263 mando=C
  gk_pontos_last1                                = 5.0000
  gk_pontos_mean_3                               = 5.0000
  gk_pontos_mean_3_n_obs                         = 3.0000
  gk_pontos_mean_5                               = 4.8000
  gk_pontos_mean_5_n_obs                         = 5.0000
  gk_pontos_mean_hist                            = 3.6000
  gk_DE_mean_3                                   = 2.6667
  gk_DE_mean_3_n_obs                             = 3.0000
  gk_DE_mean_5                                   = 2.6000
  gk_DE_mean_5_n_obs                             = 5.0000
  team_gols_sofridos_mean_3                      = 0.6667
  team_gols_sofridos_mean_3_n_obs                = 3
  team_gols_sofridos_mean_5                      = 0.6000
  team_gols_sofridos_mean_5_n_obs                = 5
  team_clean_sheet_rate_mean_3                   = 0.6667
  team_clean_sheet_rate_mean_3_n_obs             = 3
  team_clean_sheet_rate_mean_5                   = 0.6000
  team_clean_sheet_rate_mean_5_n_obs             = 5
  opp_gols_marcados_mean_3                       = 1.0000
  opp_gols_marcados_mean_3_n_obs                 = 3
  opp_gols_marcados_mean_5                       = 1.6000
  opp_gols_marcados_mean_5_n_obs                 = 5
  opp_FTT_mean_3                                 = 11.3333
  opp_FTT_mean_3_n_obs                           = 3
  opp_FTT_mean_5                                 = 10.2000
  opp_FTT_mean_5_n_obs                           = 5
  --- team window k=3: rounds R16..R18 (encontradas=3, n_obs=3)
      R16  gols_sofridos=2  clean_sheet=0
      R17  gols_sofridos=0  clean_sheet=1
      R18  gols_sofridos=0  clean_sheet=1
  --- team window k=5: rounds R14..R18 (encontradas=5, n_obs=5)
      R14  gols_sofridos=1  clean_sheet=0
      R15  gols_sofridos=0  clean_sheet=1
      R16  gols_sofridos=2  clean_sheet=0
      R17  gols_sofridos=0  clean_sheet=1
      R18  gols_sofridos=0  clean_sheet=1
  --- opp window k=3: rounds R16..R18 (encontradas=3)
      R16  gols_marcados=1  FTT=12
      R17  gols_marcados=1  FTT=14
      R18  gols_marcados=1  FTT=8
  --- opp window k=5: rounds R14..R18 (encontradas=5)
      R14  gols_marcados=2  FTT=4
      R15  gols_marcados=3  FTT=13
      R16  gols_marcados=1  FTT=12
      R17  gols_marcados=1  FTT=14
      R18  gols_marcados=1  FTT=8
  --- gk window k=3: rounds R16..R18 (participações encontradas=3)
      R16  pontos_num=1.0  scout_DE=15.0000  
      R17  pontos_num=9.0  scout_DE=20.0000  
      R18  pontos_num=5.0  scout_DE=20.0000  
  --- gk window k=5: rounds R14..R18 (participações encontradas=3)
      R16  pontos_num=1.0  scout_DE=15.0000  
      R17  pontos_num=9.0  scout_DE=20.0000  
      R18  pontos_num=5.0  scout_DE=20.0000  

### 2024 R38 — atleta_id=37656 clube=266 adv=275 mando=F
  gk_pontos_last1                                = 8.0000
  gk_pontos_mean_3                               = 6.0000
  gk_pontos_mean_3_n_obs                         = 3.0000
  gk_pontos_mean_5                               = 5.4000
  gk_pontos_mean_5_n_obs                         = 5.0000
  gk_pontos_mean_hist                            = 4.6312
  gk_DE_mean_3                                   = 3.0000
  gk_DE_mean_3_n_obs                             = 3.0000
  gk_DE_mean_5                                   = 3.0000
  gk_DE_mean_5_n_obs                             = 5.0000
  team_gols_sofridos_mean_3                      = 0.3333
  team_gols_sofridos_mean_3_n_obs                = 3
  team_gols_sofridos_mean_5                      = 1.0000
  team_gols_sofridos_mean_5_n_obs                = 5
  team_clean_sheet_rate_mean_3                   = 0.6667
  team_clean_sheet_rate_mean_3_n_obs             = 3
  team_clean_sheet_rate_mean_5                   = 0.4000
  team_clean_sheet_rate_mean_5_n_obs             = 5
  opp_gols_marcados_mean_3                       = 1.3333
  opp_gols_marcados_mean_3_n_obs                 = 3
  opp_gols_marcados_mean_5                       = 1.4000
  opp_gols_marcados_mean_5_n_obs                 = 5
  opp_FTT_mean_3                                 = 14.5000
  opp_FTT_mean_3_n_obs                           = 2
  opp_FTT_mean_5                                 = 11.3333
  opp_FTT_mean_5_n_obs                           = 3
  --- team window k=3: rounds R35..R37 (encontradas=3, n_obs=3)
      R35  gols_sofridos=0  clean_sheet=1
      R36  gols_sofridos=1  clean_sheet=0
      R37  gols_sofridos=0  clean_sheet=1
  --- team window k=5: rounds R33..R37 (encontradas=5, n_obs=5)
      R33  gols_sofridos=2  clean_sheet=0
      R34  gols_sofridos=2  clean_sheet=0
      R35  gols_sofridos=0  clean_sheet=1
      R36  gols_sofridos=1  clean_sheet=0
      R37  gols_sofridos=0  clean_sheet=1
  --- opp window k=3: rounds R35..R37 (encontradas=3)
      R35  gols_marcados=1  FTT=9
      R36  gols_marcados=1  FTT=NaN
      R37  gols_marcados=2  FTT=20
  --- opp window k=5: rounds R33..R37 (encontradas=5)
      R33  gols_marcados=1  FTT=NaN
      R34  gols_marcados=2  FTT=5
      R35  gols_marcados=1  FTT=9
      R36  gols_marcados=1  FTT=NaN
      R37  gols_marcados=2  FTT=20
  --- gk window k=3: rounds R35..R37 (participações encontradas=3)
      R35  pontos_num=8.0  scout_DE=117.0000  
      R36  pontos_num=2.0  scout_DE=120.0000  
      R37  pontos_num=8.0  scout_DE=123.0000  
  --- gk window k=5: rounds R33..R37 (participações encontradas=3)
      R35  pontos_num=8.0  scout_DE=117.0000  
      R36  pontos_num=2.0  scout_DE=120.0000  
      R37  pontos_num=8.0  scout_DE=123.0000  

### 2024 R38 — atleta_id=42234 clube=283 adv=286 mando=F
  gk_pontos_last1                                = 6.5000
  gk_pontos_mean_3                               = 2.8333
  gk_pontos_mean_3_n_obs                         = 3.0000
  gk_pontos_mean_5                               = 2.9000
  gk_pontos_mean_5_n_obs                         = 5.0000
  gk_pontos_mean_hist                            = 4.2381
  gk_DE_mean_3                                   = 4.0000
  gk_DE_mean_3_n_obs                             = 3.0000
  gk_DE_mean_5                                   = 4.0000
  gk_DE_mean_5_n_obs                             = 5.0000
  team_gols_sofridos_mean_3                      = 1.3333
  team_gols_sofridos_mean_3_n_obs                = 3
  team_gols_sofridos_mean_5                      = 1.4000
  team_gols_sofridos_mean_5_n_obs                = 5
  team_clean_sheet_rate_mean_3                   = 0.0000
  team_clean_sheet_rate_mean_3_n_obs             = 3
  team_clean_sheet_rate_mean_5                   = 0.0000
  team_clean_sheet_rate_mean_5_n_obs             = 5
  opp_gols_marcados_mean_3                       = 2.0000
  opp_gols_marcados_mean_3_n_obs                 = 3
  opp_gols_marcados_mean_5                       = 2.0000
  opp_gols_marcados_mean_5_n_obs                 = 5
  opp_FTT_mean_3                                 = 13.0000
  opp_FTT_mean_3_n_obs                           = 2
  opp_FTT_mean_5                                 = 13.0000
  opp_FTT_mean_5_n_obs                           = 4
  --- team window k=3: rounds R35..R37 (encontradas=3, n_obs=3)
      R35  gols_sofridos=1  clean_sheet=0
      R36  gols_sofridos=1  clean_sheet=0
      R37  gols_sofridos=2  clean_sheet=0
  --- team window k=5: rounds R33..R37 (encontradas=5, n_obs=5)
      R33  gols_sofridos=1  clean_sheet=0
      R34  gols_sofridos=2  clean_sheet=0
      R35  gols_sofridos=1  clean_sheet=0
      R36  gols_sofridos=1  clean_sheet=0
      R37  gols_sofridos=2  clean_sheet=0
  --- opp window k=3: rounds R35..R37 (encontradas=3)
      R35  gols_marcados=1  FTT=14
      R36  gols_marcados=3  FTT=NaN
      R37  gols_marcados=2  FTT=12
  --- opp window k=5: rounds R33..R37 (encontradas=5)
      R33  gols_marcados=2  FTT=17
      R34  gols_marcados=2  FTT=9
      R35  gols_marcados=1  FTT=14
      R36  gols_marcados=3  FTT=NaN
      R37  gols_marcados=2  FTT=12
  --- gk window k=3: rounds R35..R37 (participações encontradas=3)
      R35  pontos_num=2.0  scout_DE=76.0000  
      R36  pontos_num=0.0  scout_DE=77.0000  
      R37  pontos_num=6.5  scout_DE=85.0000  
  --- gk window k=5: rounds R33..R37 (participações encontradas=4)
      R33  pontos_num=5.0  scout_DE=73.0000  
      R35  pontos_num=2.0  scout_DE=76.0000  
      R36  pontos_num=0.0  scout_DE=77.0000  
      R37  pontos_num=6.5  scout_DE=85.0000  

### 2025 R20 — atleta_id=37656 clube=266 adv=356 mando=C
  gk_pontos_last1                                = 2.7000
  gk_pontos_mean_3                               = 5.3667
  gk_pontos_mean_3_n_obs                         = 3.0000
  gk_pontos_mean_5                               = 3.6600
  gk_pontos_mean_5_n_obs                         = 5.0000
  gk_pontos_mean_hist                            = 4.3347
  gk_DE_mean_3                                   = 2.3333
  gk_DE_mean_3_n_obs                             = 3.0000
  gk_DE_mean_5                                   = 2.2000
  gk_DE_mean_5_n_obs                             = 5.0000
  team_gols_sofridos_mean_3                      = 2.0000
  team_gols_sofridos_mean_3_n_obs                = 3
  team_gols_sofridos_mean_5                      = 1.8000
  team_gols_sofridos_mean_5_n_obs                = 5
  team_clean_sheet_rate_mean_3                   = 0.3333
  team_clean_sheet_rate_mean_3_n_obs             = 3
  team_clean_sheet_rate_mean_5                   = 0.2000
  team_clean_sheet_rate_mean_5_n_obs             = 5
  opp_gols_marcados_mean_3                       = 1.3333
  opp_gols_marcados_mean_3_n_obs                 = 3
  opp_gols_marcados_mean_5                       = 1.6000
  opp_gols_marcados_mean_5_n_obs                 = 5
  opp_FTT_mean_3                                 = 9.0000
  opp_FTT_mean_3_n_obs                           = 3
  opp_FTT_mean_5                                 = 8.5000
  opp_FTT_mean_5_n_obs                           = 4
  --- team window k=3: rounds R17..R19 (encontradas=3, n_obs=3)
      R17  gols_sofridos=3  clean_sheet=0
      R18  gols_sofridos=0  clean_sheet=1
      R19  gols_sofridos=3  clean_sheet=0
  --- team window k=5: rounds R15..R19 (encontradas=5, n_obs=5)
      R15  gols_sofridos=1  clean_sheet=0
      R16  gols_sofridos=2  clean_sheet=0
      R17  gols_sofridos=3  clean_sheet=0
      R18  gols_sofridos=0  clean_sheet=1
      R19  gols_sofridos=3  clean_sheet=0
  --- opp window k=3: rounds R17..R19 (encontradas=3)
      R17  gols_marcados=3  FTT=16
      R18  gols_marcados=1  FTT=6
      R19  gols_marcados=0  FTT=5
  --- opp window k=5: rounds R15..R19 (encontradas=5)
      R15  gols_marcados=1  FTT=7
      R16  gols_marcados=3  FTT=NaN
      R17  gols_marcados=3  FTT=16
      R18  gols_marcados=1  FTT=6
      R19  gols_marcados=0  FTT=5
  --- gk window k=3: rounds R17..R19 (participações encontradas=3)
      R17  pontos_num=7.9  scout_DE=37.0000  
      R18  pontos_num=5.5  scout_DE=37.0000  
      R19  pontos_num=2.7  scout_DE=41.0000  
  --- gk window k=5: rounds R15..R19 (participações encontradas=5)
      R15  pontos_num=1.6  scout_DE=32.0000  
      R16  pontos_num=0.6  scout_DE=34.0000  
      R17  pontos_num=7.9  scout_DE=37.0000  
      R18  pontos_num=5.5  scout_DE=37.0000  
      R19  pontos_num=2.7  scout_DE=41.0000  

### 2025 R20 — atleta_id=42234 clube=283 adv=2305 mando=F
  gk_pontos_last1                                = -1.5000
  gk_pontos_mean_3                               = 1.8000
  gk_pontos_mean_3_n_obs                         = 3.0000
  gk_pontos_mean_5                               = 5.2200
  gk_pontos_mean_5_n_obs                         = 5.0000
  gk_pontos_mean_hist                            = 4.9718
  gk_DE_mean_3                                   = 1.0000
  gk_DE_mean_3_n_obs                             = 3.0000
  gk_DE_mean_5                                   = 2.4000
  gk_DE_mean_5_n_obs                             = 5.0000
  team_gols_sofridos_mean_3                      = 1.3333
  team_gols_sofridos_mean_3_n_obs                = 3
  team_gols_sofridos_mean_5                      = 0.8000
  team_gols_sofridos_mean_5_n_obs                = 5
  team_clean_sheet_rate_mean_3                   = 0.3333
  team_clean_sheet_rate_mean_3_n_obs             = 3
  team_clean_sheet_rate_mean_5                   = 0.6000
  team_clean_sheet_rate_mean_5_n_obs             = 5
  opp_gols_marcados_mean_3                       = 1.6667
  opp_gols_marcados_mean_3_n_obs                 = 3
  opp_gols_marcados_mean_5                       = 2.0000
  opp_gols_marcados_mean_5_n_obs                 = 5
  opp_FTT_mean_3                                 = 10.0000
  opp_FTT_mean_3_n_obs                           = 3
  opp_FTT_mean_5                                 = 8.8000
  opp_FTT_mean_5_n_obs                           = 5
  --- team window k=3: rounds R17..R19 (encontradas=3, n_obs=3)
      R17  gols_sofridos=2  clean_sheet=0
      R18  gols_sofridos=0  clean_sheet=1
      R19  gols_sofridos=2  clean_sheet=0
  --- team window k=5: rounds R15..R19 (encontradas=5, n_obs=5)
      R15  gols_sofridos=0  clean_sheet=1
      R16  gols_sofridos=0  clean_sheet=1
      R17  gols_sofridos=2  clean_sheet=0
      R18  gols_sofridos=0  clean_sheet=1
      R19  gols_sofridos=2  clean_sheet=0
  --- opp window k=3: rounds R17..R19 (encontradas=3)
      R17  gols_marcados=1  FTT=11
      R18  gols_marcados=3  FTT=9
      R19  gols_marcados=1  FTT=10
  --- opp window k=5: rounds R15..R19 (encontradas=5)
      R15  gols_marcados=3  FTT=12
      R16  gols_marcados=2  FTT=2
      R17  gols_marcados=1  FTT=11
      R18  gols_marcados=3  FTT=9
      R19  gols_marcados=1  FTT=10
  --- gk window k=3: rounds R17..R19 (participações encontradas=3)
      R17  pontos_num=-0.7  scout_DE=50.0000  
      R18  pontos_num=7.6  scout_DE=52.0000  
      R19  pontos_num=-1.5  scout_DE=52.0000  
  --- gk window k=5: rounds R15..R19 (participações encontradas=4)
      R16  pontos_num=8.9  scout_DE=49.0000  
      R17  pontos_num=-0.7  scout_DE=50.0000  
      R18  pontos_num=7.6  scout_DE=52.0000  
      R19  pontos_num=-1.5  scout_DE=52.0000  

### 2025 R38 — atleta_id=37656 clube=266 adv=265 mando=C
  gk_pontos_last1                                = 4.2000
  gk_pontos_mean_3                               = 6.0333
  gk_pontos_mean_3_n_obs                         = 3.0000
  gk_pontos_mean_5                               = 6.5000
  gk_pontos_mean_5_n_obs                         = 5.0000
  gk_pontos_mean_hist                            = 4.4791
  gk_DE_mean_3                                   = 2.3333
  gk_DE_mean_3_n_obs                             = 3.0000
  gk_DE_mean_5                                   = 3.0000
  gk_DE_mean_5_n_obs                             = 5.0000
  team_gols_sofridos_mean_3                      = 0.3333
  team_gols_sofridos_mean_3_n_obs                = 3
  team_gols_sofridos_mean_5                      = 0.4000
  team_gols_sofridos_mean_5_n_obs                = 5
  team_clean_sheet_rate_mean_3                   = 0.6667
  team_clean_sheet_rate_mean_3_n_obs             = 3
  team_clean_sheet_rate_mean_5                   = 0.6000
  team_clean_sheet_rate_mean_5_n_obs             = 5
  opp_gols_marcados_mean_3                       = 1.3333
  opp_gols_marcados_mean_3_n_obs                 = 3
  opp_gols_marcados_mean_5                       = 1.6000
  opp_gols_marcados_mean_5_n_obs                 = 5
  opp_FTT_mean_3                                 = 14.0000
  opp_FTT_mean_3_n_obs                           = 3
  opp_FTT_mean_5                                 = 14.0000
  opp_FTT_mean_5_n_obs                           = 5
  --- team window k=3: rounds R35..R37 (encontradas=3, n_obs=3)
      R35  gols_sofridos=0  clean_sheet=1
      R36  gols_sofridos=0  clean_sheet=1
      R37  gols_sofridos=1  clean_sheet=0
  --- team window k=5: rounds R33..R37 (encontradas=5, n_obs=5)
      R33  gols_sofridos=0  clean_sheet=1
      R34  gols_sofridos=1  clean_sheet=0
      R35  gols_sofridos=0  clean_sheet=1
      R36  gols_sofridos=0  clean_sheet=1
      R37  gols_sofridos=1  clean_sheet=0
  --- opp window k=3: rounds R35..R37 (encontradas=3)
      R35  gols_marcados=1  FTT=11
      R36  gols_marcados=1  FTT=9
      R37  gols_marcados=2  FTT=22
  --- opp window k=5: rounds R33..R37 (encontradas=5)
      R33  gols_marcados=2  FTT=12
      R34  gols_marcados=2  FTT=16
      R35  gols_marcados=1  FTT=11
      R36  gols_marcados=1  FTT=9
      R37  gols_marcados=2  FTT=22
  --- gk window k=3: rounds R35..R37 (participações encontradas=3)
      R35  pontos_num=7.6  scout_DE=75.0000  
      R36  pontos_num=6.3  scout_DE=76.0000  
      R37  pontos_num=4.2  scout_DE=80.0000  
  --- gk window k=5: rounds R33..R37 (participações encontradas=5)
      R33  pontos_num=11.5  scout_DE=70.0000  
      R34  pontos_num=2.9  scout_DE=73.0000  
      R35  pontos_num=7.6  scout_DE=75.0000  
      R36  pontos_num=6.3  scout_DE=76.0000  
      R37  pontos_num=4.2  scout_DE=80.0000  

### 2025 R38 — atleta_id=105949 clube=285 adv=280 mando=C
  gk_pontos_last1                                = 2.7000
  gk_pontos_mean_3                               = 1.0667
  gk_pontos_mean_3_n_obs                         = 3.0000
  gk_pontos_mean_5                               = 0.2800
  gk_pontos_mean_5_n_obs                         = 5.0000
  gk_pontos_mean_hist                            = 3.2486
  gk_DE_mean_3                                   = 3.0000
  gk_DE_mean_3_n_obs                             = 3.0000
  gk_DE_mean_5                                   = 2.6000
  gk_DE_mean_5_n_obs                             = 5.0000
  team_gols_sofridos_mean_3                      = 3.0000
  team_gols_sofridos_mean_3_n_obs                = 3
  team_gols_sofridos_mean_5                      = 2.4000
  team_gols_sofridos_mean_5_n_obs                = 5
  team_clean_sheet_rate_mean_3                   = 0.0000
  team_clean_sheet_rate_mean_3_n_obs             = 3
  team_clean_sheet_rate_mean_5                   = 0.0000
  team_clean_sheet_rate_mean_5_n_obs             = 5
  opp_gols_marcados_mean_3                       = 0.6667
  opp_gols_marcados_mean_3_n_obs                 = 3
  opp_gols_marcados_mean_5                       = 1.4000
  opp_gols_marcados_mean_5_n_obs                 = 5
  opp_FTT_mean_3                                 = 5.0000
  opp_FTT_mean_3_n_obs                           = 1
  opp_FTT_mean_5                                 = 5.0000
  opp_FTT_mean_5_n_obs                           = 2
  --- team window k=3: rounds R35..R37 (encontradas=3, n_obs=3)
      R35  gols_sofridos=1  clean_sheet=0
      R36  gols_sofridos=5  clean_sheet=0
      R37  gols_sofridos=3  clean_sheet=0
  --- team window k=5: rounds R33..R37 (encontradas=5, n_obs=5)
      R33  gols_sofridos=2  clean_sheet=0
      R34  gols_sofridos=1  clean_sheet=0
      R35  gols_sofridos=1  clean_sheet=0
      R36  gols_sofridos=5  clean_sheet=0
      R37  gols_sofridos=3  clean_sheet=0
  --- opp window k=3: rounds R35..R37 (encontradas=3)
      R35  gols_marcados=0  FTT=5
      R36  gols_marcados=0  FTT=NaN
      R37  gols_marcados=2  FTT=NaN
  --- opp window k=5: rounds R33..R37 (encontradas=5)
      R33  gols_marcados=1  FTT=5
      R34  gols_marcados=4  FTT=NaN
      R35  gols_marcados=0  FTT=5
      R36  gols_marcados=0  FTT=NaN
      R37  gols_marcados=2  FTT=NaN
  --- gk window k=3: rounds R35..R37 (participações encontradas=3)
      R35  pontos_num=0.3  scout_DE=38.0000  
      R36  pontos_num=0.2  scout_DE=42.0000  
      R37  pontos_num=2.7  scout_DE=46.0000  
  --- gk window k=5: rounds R33..R37 (participações encontradas=3)
      R35  pontos_num=0.3  scout_DE=38.0000  
      R36  pontos_num=0.2  scout_DE=42.0000  
      R37  pontos_num=2.7  scout_DE=46.0000  

### 2026 R20 — atleta_id=37656 clube=266 adv=284 mando=F
  gk_pontos_last1                                = 1.6000
  gk_pontos_mean_3                               = 5.5000
  gk_pontos_mean_3_n_obs                         = 3.0000
  gk_pontos_mean_5                               = 4.5800
  gk_pontos_mean_5_n_obs                         = 5.0000
  gk_pontos_mean_hist                            = 4.5430
  gk_DE_mean_3                                   = 63.3333
  gk_DE_mean_3_n_obs                             = 3.0000
  gk_DE_mean_5                                   = 58.6000
  gk_DE_mean_5_n_obs                             = 5.0000
  team_gols_sofridos_mean_3                      = 1.0000
  team_gols_sofridos_mean_3_n_obs                = 3
  team_gols_sofridos_mean_5                      = 1.2000
  team_gols_sofridos_mean_5_n_obs                = 5
  team_clean_sheet_rate_mean_3                   = 0.0000
  team_clean_sheet_rate_mean_3_n_obs             = 3
  team_clean_sheet_rate_mean_5                   = 0.0000
  team_clean_sheet_rate_mean_5_n_obs             = 5
  opp_gols_marcados_mean_3                       = 1.6667
  opp_gols_marcados_mean_3_n_obs                 = 3
  opp_gols_marcados_mean_5                       = 1.2000
  opp_gols_marcados_mean_5_n_obs                 = 5
  opp_FTT_mean_3                                 = 5.3333
  opp_FTT_mean_3_n_obs                           = 3
  opp_FTT_mean_5                                 = 4.8000
  opp_FTT_mean_5_n_obs                           = 5
  --- team window k=3: rounds R17..R19 (encontradas=3, n_obs=3)
      R17  gols_sofridos=1  clean_sheet=0
      R18  gols_sofridos=1  clean_sheet=0
      R19  gols_sofridos=1  clean_sheet=0
  --- team window k=5: rounds R15..R19 (encontradas=5, n_obs=5)
      R15  gols_sofridos=2  clean_sheet=0
      R16  gols_sofridos=1  clean_sheet=0
      R17  gols_sofridos=1  clean_sheet=0
      R18  gols_sofridos=1  clean_sheet=0
      R19  gols_sofridos=1  clean_sheet=0
  --- opp window k=3: rounds R17..R19 (encontradas=3)
      R17  gols_marcados=3  FTT=10
      R18  gols_marcados=1  FTT=6
      R19  gols_marcados=1  FTT=0
  --- opp window k=5: rounds R15..R19 (encontradas=5)
      R15  gols_marcados=0  FTT=5
      R16  gols_marcados=1  FTT=3
      R17  gols_marcados=3  FTT=10
      R18  gols_marcados=1  FTT=6
      R19  gols_marcados=1  FTT=0
  --- gk window k=3: rounds R17..R19 (participações encontradas=2)
      R17  pontos_num=8.1  scout_DE=65.0000  
      R18  pontos_num=1.6  scout_DE=67.0000  
  --- gk window k=5: rounds R15..R19 (participações encontradas=4)
      R15  pontos_num=-0.7  scout_DE=52.0000  
      R16  pontos_num=6.8  scout_DE=58.0000  
      R17  pontos_num=8.1  scout_DE=65.0000  
      R18  pontos_num=1.6  scout_DE=67.0000  

### 2026 R20 — atleta_id=51413 clube=2305 adv=267 mando=F
  gk_pontos_last1                                = 0.3000
  gk_pontos_mean_3                               = 2.5000
  gk_pontos_mean_3_n_obs                         = 3.0000
  gk_pontos_mean_5                               = 2.4600
  gk_pontos_mean_5_n_obs                         = 5.0000
  gk_pontos_mean_hist                            = 4.6039
  gk_DE_mean_3                                   = 50.0000
  gk_DE_mean_3_n_obs                             = 3.0000
  gk_DE_mean_5                                   = 47.6000
  gk_DE_mean_5_n_obs                             = 5.0000
  team_gols_sofridos_mean_3                      = 0.6667
  team_gols_sofridos_mean_3_n_obs                = 3
  team_gols_sofridos_mean_5                      = 1.2000
  team_gols_sofridos_mean_5_n_obs                = 5
  team_clean_sheet_rate_mean_3                   = 0.3333
  team_clean_sheet_rate_mean_3_n_obs             = 3
  team_clean_sheet_rate_mean_5                   = 0.2000
  team_clean_sheet_rate_mean_5_n_obs             = 5
  opp_gols_marcados_mean_3                       = 0.0000
  opp_gols_marcados_mean_3_n_obs                 = 3
  opp_gols_marcados_mean_5                       = 0.4000
  opp_gols_marcados_mean_5_n_obs                 = 5
  opp_FTT_mean_3                                 = 8.3333
  opp_FTT_mean_3_n_obs                           = 3
  opp_FTT_mean_5                                 = 9.4000
  opp_FTT_mean_5_n_obs                           = 5
  --- team window k=3: rounds R17..R19 (encontradas=3, n_obs=3)
      R17  gols_sofridos=0  clean_sheet=1
      R18  gols_sofridos=1  clean_sheet=0
      R19  gols_sofridos=1  clean_sheet=0
  --- team window k=5: rounds R15..R19 (encontradas=5, n_obs=5)
      R15  gols_sofridos=1  clean_sheet=0
      R16  gols_sofridos=3  clean_sheet=0
      R17  gols_sofridos=0  clean_sheet=1
      R18  gols_sofridos=1  clean_sheet=0
      R19  gols_sofridos=1  clean_sheet=0
  --- opp window k=3: rounds R17..R19 (encontradas=3)
      R17  gols_marcados=0  FTT=10
      R18  gols_marcados=0  FTT=15
      R19  gols_marcados=0  FTT=0
  --- opp window k=5: rounds R15..R19 (encontradas=5)
      R15  gols_marcados=1  FTT=12
      R16  gols_marcados=1  FTT=10
      R17  gols_marcados=0  FTT=10
      R18  gols_marcados=0  FTT=15
      R19  gols_marcados=0  FTT=0
  --- gk window k=3: rounds R17..R19 (participações encontradas=2)
      R17  pontos_num=6.3  scout_DE=50.0000  
      R18  pontos_num=0.3  scout_DE=51.0000  
  --- gk window k=5: rounds R15..R19 (participações encontradas=4)
      R15  pontos_num=4.2  scout_DE=46.0000  
      R16  pontos_num=0.9  scout_DE=49.0000  
      R17  pontos_num=6.3  scout_DE=50.0000  
      R18  pontos_num=0.3  scout_DE=51.0000  