# Confronto v1 — features construídas

- Gerado em: 2026-09-20T12:04:00
- Fonte: Football-Data `AvgCH/AvgCD/AvgCA` (closing odds)
- Fontes auxiliares: Bzzoiro (rodada), Cartola (clube_nome)

## Cobertura

- Cartola total: 77253
- Com odds: 77023 (99.70%)
- Sem odds: 230

## Transformações

- `q_x = 1/odds_x`
- `p_x = q_x / (q_home+q_draw+q_away)`
- `favoritismo_home_away = log(p_home/p_away)`
- `mando = 1` se mandante, `0` se visitante

## Overround

- min=1.0465
- mediana=1.0664
- media=1.0655
- max=1.0919

## Parquet canônico

- SHA-256 inalterado: True
- hash: 15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4
