# Auditoria das odds históricas (Football-Data)

- Arquivo: `data/raw/odds/football_data_bra.csv`
- Linhas: 5587
- Temporadas: [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
- Data: 2026-09-20T11:00:04

## Fonte

Football-Data.co.uk — `BRA.csv` (Brasileirão Série A, 2012–2026).
Colunas usadas: `AvgCH`, `AvgCD`, `AvgCA` = **odds de fechamento (closing)**.

## Duplicatas (Season, Date, Home, Away)

- `0`

## Cobertura por temporada

```
 Season  n_partidas  n_completas  cobertura
   2012         380          380        1.0
   2013         380          380        1.0
   2014         380          380        1.0
   2015         380          380        1.0
   2016         380          380        1.0
   2017         380          380        1.0
   2018         380          380        1.0
   2019         380          380        1.0
   2020         380          380        1.0
   2021         380          380        1.0
   2022         380          380        1.0
   2023         380          380        1.0
   2024         380          380        1.0
   2025         380          380        1.0
   2026         267          267        1.0
```

## Diagnóstico global

- Nulos AvgCH: 0
- Nulos AvgCD: 0
- Nulos AvgCA: 0
- <=1 AvgCH: 0
- <=1 AvgCD: 0
- <=1 AvgCA: 0
- AvgCH max: 20.23
- AvgCA max: 26.58

## Clubes distintos

- total: 38

- America MG
- Athletico-PR
- Atletico GO
- Atletico-MG
- Avai
- Bahia
- Botafogo RJ
- Bragantino
- CSA
- Ceara
- Chapecoense-SC
- Corinthians
- Coritiba
- Criciuma
- Cruzeiro
- Cuiaba
- Figueirense
- Flamengo RJ
- Fluminense
- Fortaleza
- Goias
- Gremio
- Internacional
- Joinville
- Juventude
- Mirassol
- Nautico
- Palmeiras
- Parana
- Ponte Preta
- Portuguesa
- Remo
- Santa Cruz
- Santos
- Sao Paulo
- Sport Recife
- Vasco
- Vitoria
