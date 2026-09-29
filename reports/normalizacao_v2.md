# Normalizacao v2 — caRtola

- **Inicio:** 2026-09-20T06:24:49
- **Fim:** 2026-09-20T06:24:56
- **Duracao:** 6.6s

## Arquivos processados por temporada

- 2024: 38 CSV(s)
- 2025: 38 CSV(s)
- 2026: 27 CSV(s)

## Linhas lidas por temporada

- 2024: 28499  (esperado 28499)  OK
- 2025: 28589  (esperado 28589)  OK
- 2026: 20165  (esperado 20165)  OK
- **Total lido:** 77253  (esperado 77253)  OK

## Linhas no Parquet

- **Total:** 77253  OK

## DP / GC estruturalmente ausentes

- Linhas com `scout_DP` NaN (coluna ausente na rodada): 4091
- Linhas com `scout_GC` NaN (coluna ausente na rodada): 2696
- NaN em `scout_DP` (todas as causas): 76733
- Todas as linhas de rodadas sem coluna DP estao NaN: True
- Distribuicao de NaN em scout_GC confere: True

Rodadas em que a coluna **nao existia** no RAW:

- DP: {'2024': [1, 2, 3], '2025': [1], '2026': [1, 2]}
- GC: {'2024': [1, 2, 3], '2025': [1], '2026': []}

## Distribuicao de entrou_em_campo

- False: 45438
- True: 31815

## Atletas distintos por temporada

- 2024: 947
- 2025: 980
- 2026: 970

## Chaves duplicadas (temporada, rodada, atleta_id)

- Duplicatas: 0  OK

## pontos_num nulos

- 0  OK

## Rodadas presentes por temporada

- 2024: 38 rodadas (min=1, max=38)  OK
- 2025: 38 rodadas (min=1, max=38)  OK
- 2026: 27 rodadas (min=1, max=27)  OK

- Todas as rodadas esperadas presentes: True

## Colunas finais do canonico

- temporada
- atleta_id
- apelido
- apelido_abreviado
- nome
- slug
- clube_id
- clube_nome
- posicao_id
- status_id
- entrou_em_campo
- media_num
- variacao_num
- preco_num
- jogos_num
- pontos_num
- rodada
- scout_A
- scout_CA
- scout_CV
- scout_DE
- scout_DS
- scout_FC
- scout_FD
- scout_FF
- scout_FS
- scout_FT
- scout_G
- scout_I
- scout_PC
- scout_PS
- scout_PP
- scout_SG
- scout_V
- scout_DP
- scout_GC
- arquivo_origem
- linha_origem

## Amostra de 5 linhas

```
       temporada  atleta_id     apelido apelido_abreviado                                  nome        slug  clube_id clube_nome  posicao_id  status_id  entrou_em_campo  media_num  variacao_num  preco_num  jogos_num  pontos_num  rodada  scout_A  scout_CA  scout_CV  scout_DE  scout_DS  scout_FC  scout_FD  scout_FF  scout_FS  scout_FT  scout_G  scout_I  scout_PC  scout_PS  scout_PP  scout_SG  scout_V  scout_DP  scout_GC      arquivo_origem  linha_origem
67368       2026      84268    Georgemy          Georgemy                    Georgemy Gonçalves    georgemy      2305        MIR           1          6            False       0.00          0.00       3.00          0         0.0      15      NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN      NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN  2026/rodada-15.csv           104
72035       2026      94214  Léo Duarte         L. Duarte       Leonardo Campos Duarte da Silva  leo-duarte       282        CAM           3          5            False       0.00          0.00       5.00          0         0.0      21      NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN      NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN  2026/rodada-21.csv           188
44774       2025     107554    Natanael          Natanael             Natanael Moreira Milouski    natanael       282        CAM           2          2             True       5.17          0.50       8.56         21         5.7      23      NaN       1.0       NaN       NaN      35.0      18.0       4.0       8.0      17.0       NaN      1.0      2.0       NaN       NaN       NaN       7.0      NaN       NaN       NaN  2025/rodada-23.csv           437
10137       2024      91706      Marlon            Marlon               Marlon Rodrigues Xavier      marlon       283   Cruzeiro           2          7             True       3.46          0.51       8.91         10         5.6      15      NaN       NaN       2.0       NaN      17.0      16.0       1.0       1.0       6.0       NaN      NaN      NaN       NaN       NaN       NaN       4.0      NaN       NaN       NaN  2024/rodada-15.csv           219
30998       2025     115404   Rwan Cruz           R. Cruz  Rwan Philipe Rodrigues de Souza Cruz   rwan-cruz       263        BOT           5          6            False       0.00          0.00       4.00          0         0.0       4      NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN      NaN       NaN       NaN       NaN       NaN      NaN       NaN       NaN   2025/rodada-4.csv           519
```

## Integridade dos RAWs

- RAWs inalterados (SHA-256 antes == depois): **True**

## Validacao do Parquet

- Existe: True
- Reabre com pandas/pyarrow: True
