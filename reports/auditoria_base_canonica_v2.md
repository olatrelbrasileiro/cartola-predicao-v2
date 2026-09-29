# Auditoria da base canonica v2 — caRtola

- **Inicio:** 2026-09-20T06:33:01
- **Fim:** 2026-09-20T06:33:07
- **Duracao:** 6.3s

## 3.1 Alvo (`pontos_num`)

Ver `reports/auditoria_alvo.csv`.

Amostra (temporada):

```
             grupo  temporada  count      mean  median      std  min  max  p01  p05  p25  p50  p75    p95  p99  n_zeros  n_neg  n_pos  rodada  posicao_id
         temporada     2024.0  28499  1.346791     0.0 2.848433 -6.0 28.3 -1.1  0.0  0.0  0.0  1.4  7.841 12.4    17660   1211   9628     NaN         NaN
         temporada     2025.0  28589  1.459945     0.0 3.071350 -4.7 33.1 -1.1  0.0  0.0  0.0  1.5  8.300 13.5    17952   1129   9508     NaN         NaN
         temporada     2026.0  20165  1.516560     0.0 3.093300 -6.0 27.6 -1.1  0.0  0.0  0.0  1.7  8.494 13.7    12280    851   7034     NaN         NaN
  A_nao_participou        NaN  45438 -0.003017     0.0 0.070619 -3.0  0.0  0.0  0.0  0.0  0.0  0.0  0.000  0.0    45330    108      0     NaN         NaN
   B_entrou_e_zero        NaN   2562  0.000000     0.0 0.000000  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.000  0.0     2562      0      0     NaN         NaN
C_entrou_e_pontuou        NaN  29253  3.788983     2.8 3.845827 -6.0 33.1 -1.8 -0.6  1.1  2.8  5.7 11.200 16.4        0   3083  26170     NaN         NaN
```

## 3.2 Auditoria temporal

Ver `reports/auditoria_temporal.csv`.

Contagem de rodadas por classificacao:

temporada  classificacao    
2024       NORMAL               38
2025       NORMAL               37
           POSSIVEL_SNAPSHOT     1
2026       NORMAL               27

Rodadas nao-NORMAL:

```
 temporada  rodada  n_jogadores  n_entrou  n_nao_entrou  prop_entrou  pts_mean  pts_median  pts_zeros  pts_min  pts_max  prop_zeros     classificacao
      2025       1          619         0           619          0.0       0.0         0.0        619      0.0      0.0         1.0 POSSIVEL_SNAPSHOT
```

## 3.3 `entrou_em_campo` vs `jogos_num`

- n_linhas_analisadas: 74318
- caso_entrou_e_nao_incrementou: 0
- caso_nao_entrou_e_incrementou: 59
- caso_jogos_num_retrocedeu: 0

Exemplos de 'entrou e nao incrementou' (ate 5):

```
Empty DataFrame
Columns: []
Index: []
```

Exemplos de 'nao entrou e incrementou' (ate 5):

```
 temporada  rodada  atleta_id           apelido  jogos_prev  jogos_num  delta_jogos  entrou_em_campo  pontos_num
      2025       2      51772   Everton Ribeiro         0.0          1          1.0            False         0.0
      2025       2      68899 Philippe Coutinho         0.0          1          1.0            False         0.0
      2025       2      68952           Marinho         0.0          1          1.0            False         0.0
      2025       2      70449           Gabriel         0.0          1          1.0            False         0.0
      2025       2      70916     Diogo Barbosa         0.0          1          1.0            False         0.0
```

Exemplos de 'jogos_num retrocedeu' (ate 5):

```
Empty DataFrame
Columns: []
Index: []
```

## 3.4 Scouts

Ver `reports/auditoria_scouts.csv`.

Resumo de NaN estrutural vs nao-estrutural:

- `scout_DP`: {'linhas_ausencia_estrutural': 4091, 'linhas_nan_total': 76733, 'nan_nao_estrutural': 72642}
- `scout_GC`: {'linhas_ausencia_estrutural': 2696, 'linhas_nan_total': 76403, 'nan_nao_estrutural': 73707}

NaN por scout (agregado por temporada):

```
scout  temporada  n_linhas  n_nan  n_zero  n_positivo  prop_nan
    A       2024     28499  22180       0        6319  0.778273
    A       2025     28589  22718       0        5871  0.794641
    A       2026     20165  16383       0        3782  0.812447
   CA       2024     28499  14821       0       13678  0.520053
   CA       2025     28589  15734       0       12855  0.550352
   CA       2026     20165  12451       0        7714  0.617456
   CV       2024     28499  26316       0        2183  0.923401
   CV       2025     28589  27216       0        1373  0.951975
   CV       2026     20165  19222       0         943  0.953236
   DE       2024     28499  27247       0        1252  0.956069
   DE       2025     28589  27291       0        1298  0.954598
   DE       2026     20165  19333       0         832  0.958740
   DP       2024     28499  28299       0         200  0.992982
   DP       2025     28589  28367       0         222  0.992235
   DP       2026     20165  20067       0          98  0.995140
   DS       2024     28499  11349       0       17150  0.398224
   DS       2025     28589  11124       0       17465  0.389101
   DS       2026     20165   8688       0       11477  0.430846
   FC       2024     28499  11124       0       17375  0.390329
   FC       2025     28589  10984       0       17605  0.384204
   FC       2026     20165   8512       0       11653  0.422118
   FD       2024     28499  16402       0       12097  0.575529
   FD       2025     28589  16842       0       11747  0.589108
   FD       2026     20165  13072       0        7093  0.648252
   FF       2024     28499  14255       0       14244  0.500193
   FF       2025     28589  14250       0       14339  0.498443
   FF       2026     20165  10942       0        9223  0.542623
   FS       2024     28499  11057       0       17442  0.387979
   FS       2025     28589  10745       0       17844  0.375844
   FS       2026     20165   8710       0       11455  0.431937
   FT       2024     28499  25296       0        3203  0.887610
   FT       2025     28589  25221       0        3368  0.882192
   FT       2026     20165  18434       0        1731  0.914158
    G       2024     28499  21024       0        7475  0.737710
    G       2025     28589  21444       0        7145  0.750079
    G       2026     20165  15839       0        4326  0.785470
   GC       2024     28499  28204       0         295  0.989649
   GC       2025     28589  28288       0         301  0.989471
   GC       2026     20165  19911       0         254  0.987404
    I       2024     28499  21066       0        7433  0.739184
    I       2025     28589  20723       0        7866  0.724859
    I       2026     20165  15960       0        4205  0.791470
   PC       2024     28499  26944       0        1555  0.945437
   PC       2025     28589  27091       0        1498  0.947602
   PC       2026     20165  19360       0         805  0.960079
   PP       2024     28499  28148       0         351  0.987684
   PP       2025     28589  28194       0         395  0.986183
   PP       2026     20165  20027       0         138  0.993156
   PS       2024     28499  27477       0        1022  0.964139
   PS       2025     28589  27486       0        1103  0.961419
   PS       2026     20165  19584       0         581  0.971188
   SG       2024     28499  22417       0        6082  0.786589
   SG       2025     28589  21989       0        6600  0.769142
   SG       2026     20165  16462       0        3703  0.816365
    V       2024     28499  27827       0         672  0.976420
    V       2025     28589  27919       0         670  0.976564
    V       2026     20165  19695       0         470  0.976692
```

## 3.5 Outliers de pontos

- min_global: -6.0
- max_global: 33.1
- n_lt_menos5: 3
- n_gt_20: 84
- n_gt_30: 2
- n_gt_40: 0

Top 20 maiores e menores (ver CSV completo em `reports/auditoria_outliers.csv`):

```
 temporada  rodada  atleta_id         apelido  posicao_id  clube_id  clube_nome  pontos_num  entrou_em_campo  jogos_num  scout_G  scout_A  scout_SG  scout_DS  scout_FC  scout_FS  scout_CA  scout_CV  scout_GC  scout_PC  scout_PS  scout_PP  scout_FT  scout_FD  scout_FF  scout_I  scout_V  scout_DE  scout_G
      2024       3      70449         Gabriel           1       286   Juventude        -6.0             True          3      NaN      NaN       1.0       NaN       NaN       2.0       2.0       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN      NaN       4.0      NaN
      2026       1     103056  Gabriel Brazão           1       277         SAN        -6.0             True          1      NaN      NaN       NaN       NaN       NaN       NaN       1.0       NaN       NaN       1.0       NaN       NaN       NaN       NaN       NaN      NaN      NaN       NaN      NaN
      2024      29      72294         Everson           1       282 Atlético-MG        -5.3             True         17      NaN      NaN       5.0       NaN       1.0       4.0       5.0       1.0       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN      NaN      40.0      NaN
      2026      12     100804         Tchamba           3       364         REM        -4.9             True          8      NaN      NaN       1.0      11.0       6.0       2.0       NaN       NaN       2.0       NaN       NaN       NaN       NaN       NaN       1.0      1.0      NaN       NaN      NaN
      2026       4     111246   Alix Vinicius           3       280         RBB        -4.8             True          3      NaN      NaN       2.0       NaN      10.0       NaN       1.0       1.0       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN      NaN       NaN      NaN
      2025      12      81881    João Ricardo           1       356         FOR        -4.7             True         11      NaN      NaN       4.0       NaN       NaN       NaN       NaN       NaN       1.0       NaN       NaN       NaN       NaN       NaN       NaN      NaN      NaN      41.0      NaN
      2025      30     115594    Marcos Paulo           2       286         JUV        -4.5             True         13      NaN      NaN       2.0      14.0      20.0       4.0       5.0       1.0       NaN       1.0       NaN       NaN       NaN       2.0       2.0      NaN      NaN       NaN      NaN
      2024      28      42500          Fagner           2       264 Corinthians        -4.3             True         13      NaN      1.0       5.0      13.0       3.0       9.0       1.0       1.0       NaN       1.0       NaN       NaN       1.0       1.0       NaN      NaN      NaN       NaN      NaN
      2026      13     101393    Eduardo Doma           3       315         CHA        -4.3             True         12      1.0      NaN       1.0      15.0       9.0       2.0       2.0       1.0       NaN       1.0       NaN       NaN       NaN       1.0       5.0      NaN      NaN       NaN      1.0
      2025      21     104118   Lucas Arcanjo           1       287         VIT        -4.1             True         20      NaN      NaN       4.0       NaN       NaN       3.0       2.0       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN      NaN      75.0      NaN
      2024       9     101995          Brítez           3       356   Fortaleza        -4.0             True          6      NaN      NaN       1.0       3.0       6.0       2.0       4.0       1.0       NaN       NaN       NaN       NaN       NaN       1.0       2.0      1.0      NaN       NaN      NaN
      2024      16     111246   Alix Vinicius           3       373 Atlético-GO        -4.0             True         10      NaN      NaN       1.0       4.0      10.0       2.0       NaN       1.0       1.0       NaN       NaN       NaN       NaN       3.0       4.0      2.0      NaN       NaN      NaN
      2024      35      38877         Mariano           2       282 Atlético-MG        -4.0             True         10      NaN      NaN       2.0       9.0       6.0       6.0       4.0       1.0       NaN       1.0       NaN       NaN       NaN       NaN       NaN      NaN      NaN       NaN      NaN
      2026      24      87008      Léo Jardim           1       267         VAS        -4.0             True         22      NaN      NaN       2.0       1.0       1.0       2.0       1.0       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN      NaN      62.0      NaN
      2026      26      97899   Matheus Bahia           2       285         INT        -4.0             True         17      NaN      2.0       4.0      29.0      18.0      19.0       3.0       1.0       NaN       1.0       NaN       NaN       NaN       1.0       1.0      3.0      NaN       NaN      NaN
      2024       1      94845       Battaglia           4       282 Atlético-MG        -3.9             True          1      NaN      NaN       NaN       NaN       3.0       NaN       NaN       1.0       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN      NaN       NaN      NaN
      2024      34     101752       Claudinho           2       288    Criciúma        -3.9             True         30      2.0      3.0       4.0      49.0      13.0       7.0       3.0       1.0       NaN       NaN       NaN       NaN       1.0       3.0       8.0      4.0      NaN       NaN      2.0
      2026      13     145744           André           4       264         COR        -3.9             True         10      1.0      NaN       NaN      16.0      12.0      15.0       1.0       2.0       NaN       NaN       NaN       NaN       NaN       1.0       4.0      NaN      NaN       NaN      1.0
      2026      24     101405       Wallisson           4      2305         MIR        -3.9             True         15      NaN      NaN       NaN      23.0      21.0      11.0       3.0       1.0       NaN       NaN       NaN       NaN       NaN       NaN       4.0      1.0      NaN       NaN      NaN
      2025      17     121866          Hudson           4       286         JUV        -3.7             True          3      NaN      NaN       NaN       3.0       6.0       NaN       2.0       1.0       NaN       NaN       NaN       NaN       NaN       NaN       NaN      NaN      NaN       NaN      NaN
      2025      21      98873     Samuel Lino           5       262         FLA        33.1             True          4      2.0      4.0       NaN       5.0       4.0       3.0       NaN       NaN       NaN       NaN       NaN       NaN       NaN       2.0       7.0      1.0      NaN       NaN      2.0
      2025      21      94583           Pedro           5       262         FLA        31.9             True         14     10.0      3.0       NaN       7.0       8.0       7.0       3.0       NaN       NaN       NaN       1.0       1.0       1.0      11.0       2.0      1.0      NaN       NaN     10.0
      2024      24     128915         Estêvão           5       275   Palmeiras        28.3             True         20      7.0      7.0       NaN      23.0      19.0      19.0       2.0       NaN       NaN       NaN       1.0       NaN       1.0      13.0      16.0      3.0      NaN       NaN      7.0
      2026      19     123101        Leozinho           5       293         CAP        27.6             True          6      2.0      NaN       NaN       6.0       3.0       2.0       1.0       NaN       NaN       NaN       NaN       NaN       1.0       2.0       2.0      NaN      NaN       NaN      2.0
      2024      30      94509   Raphael Veiga           4       275   Palmeiras        27.2             True         26      8.0      3.0       NaN      18.0      13.0      38.0       3.0       1.0       NaN       NaN       NaN       1.0       1.0      22.0      31.0      4.0      NaN       NaN      8.0
      2025       8     105568          Lucero           5       356         FOR        26.5             True          7      3.0      1.0       NaN       4.0       8.0       5.0       2.0       NaN       NaN       NaN       NaN       1.0       1.0       1.0       2.0      2.0      NaN       NaN      3.0
      2024      10      89275         William           2       283    Cruzeiro        26.4             True          9      2.0      2.0       3.0      28.0       8.0       4.0       2.0       NaN       NaN       NaN       NaN       NaN       NaN       2.0       4.0      NaN      NaN       NaN      2.0
      2025      37      63007          Neymar           4       277         SAN        26.3             True         18      8.0      2.0       NaN       4.0      18.0      60.0       5.0       1.0       NaN       1.0       NaN       NaN       2.0      15.0      19.0      4.0      NaN       NaN      8.0
      2025      13     103445      Kaio Jorge           5       283         CRU        26.2             True         12     11.0      2.0       NaN      14.0       8.0      17.0       2.0       NaN       NaN       NaN       1.0       1.0       3.0       7.0      13.0      5.0      NaN       NaN     11.0
      2025      25      93716  Paulo Henrique           2       267         VAS        26.2             True         17      1.0      4.0       4.0      58.0      35.0      15.0       6.0       NaN       NaN       NaN       NaN       NaN       2.0       6.0       4.0      1.0      NaN       NaN      1.0
      2025       6      94583           Pedro           5       262         FLA        26.0             True          3      4.0      1.0       NaN       1.0       NaN       1.0       NaN       NaN       NaN       NaN       1.0       1.0       1.0       4.0       1.0      NaN      NaN       NaN      4.0
      2026      10     101290        Ferreira           5       276         SAO        26.0             True          7      3.0      NaN       NaN       2.0       1.0       7.0       2.0       NaN       NaN       NaN       NaN       NaN       NaN       1.0       2.0      NaN      NaN       NaN      3.0
      2026      16      92171   Arthur Cabral           5       263         BOT        26.0             True         12      7.0      1.0       NaN       7.0      18.0      15.0       2.0       NaN       NaN       NaN       NaN       NaN       NaN       5.0      11.0      1.0      NaN       NaN      7.0
      2026      18      98873     Samuel Lino           5       262         FLA        26.0             True         16      4.0      6.0       NaN      21.0      14.0      26.0       2.0       NaN       NaN       NaN       NaN       NaN       NaN       7.0       6.0      2.0      NaN       NaN      4.0
      2025      23     114802     Vitor Roque           5       275         PAL        25.9             True         20      8.0      3.0       NaN      22.0      22.0      44.0       2.0       NaN       NaN       NaN       NaN       NaN       NaN      13.0      11.0     11.0      NaN       NaN      8.0
      2025      28     114802     Vitor Roque           5       275         PAL        25.8             True         25     12.0      3.0       NaN      26.0      24.0      62.0       3.0       NaN       NaN       NaN       NaN       NaN       NaN      18.0      13.0     15.0      NaN       NaN     12.0
      2025       8      96610 Matheus Pereira           4       283         CRU        25.6             True          7      2.0      1.0       NaN      17.0      11.0      25.0       2.0       NaN       NaN       NaN       NaN       NaN       NaN       6.0       6.0      2.0      NaN       NaN      2.0
      2025      26     113103     Flaco López           5       275         PAL        25.2             True         22      8.0      1.0       NaN      24.0      34.0      37.0       4.0       NaN       NaN       NaN       2.0       NaN       NaN       9.0      28.0      7.0      NaN       NaN      8.0
      2025      34      95131 Gabriel Taliari           5       286         JUV        25.2             True         28      6.0      4.0       NaN      22.0      50.0      55.0       7.0       NaN       NaN       NaN       NaN       NaN       2.0       8.0      16.0      5.0      NaN       NaN      6.0
      2025      30     105563 Carlos Vinícius           5       284         GRE        25.1             True          7      6.0      NaN       NaN       1.0      13.0      16.0       2.0       1.0       NaN       NaN       NaN       NaN       NaN       2.0       3.0      4.0      NaN       NaN      6.0
```

## 3.6 Clubes / transferencias

- n_pares_temporada_atleta: 2897
- n_pares_com_troca_de_clube: 124
- prop_trocas: 0.042802899551259925

Exemplos (ate 10):

- temporada=2024 atleta_id=42234: [{'rodada': 1, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 2, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 3, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 4, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 5, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 6, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 7, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 8, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 9, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 10, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 11, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 12, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 13, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 14, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 15, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 16, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 17, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 18, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 19, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 20, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 21, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 22, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 23, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 24, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 25, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 26, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 27, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 28, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 29, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 30, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 31, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 32, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 33, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 34, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 35, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 36, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 37, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 38, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}]
- temporada=2024 atleta_id=63289: [{'rodada': 1, 'clube_id': 282, 'clube_nome': 'Atlético-MG'}, {'rodada': 2, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 3, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 4, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 5, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 6, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 7, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 8, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 9, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 10, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 11, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 12, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 13, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 14, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 15, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 16, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 17, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 18, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 19, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 20, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 21, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 22, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 23, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 24, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 25, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 26, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 27, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 28, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 29, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 30, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 31, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 32, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 33, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 34, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 35, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 36, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 37, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 38, 'clube_id': 284, 'clube_nome': 'Grêmio'}]
- temporada=2024 atleta_id=68940: [{'rodada': 1, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 2, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 3, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 4, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 5, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 6, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 7, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 8, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 9, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 10, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 11, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 12, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 13, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 14, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 15, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 16, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 17, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 18, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 19, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 20, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 21, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 22, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 23, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 24, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 25, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 26, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 27, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 28, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 29, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 30, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 31, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 32, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 33, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 34, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 35, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 36, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 37, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 38, 'clube_id': 284, 'clube_nome': 'Grêmio'}]
- temporada=2024 atleta_id=69705: [{'rodada': 1, 'clube_id': 276, 'clube_nome': 'São Paulo'}, {'rodada': 2, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 3, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 4, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 5, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 6, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 7, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 8, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 9, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 10, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 11, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 12, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 13, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 14, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 15, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 16, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 17, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 18, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 19, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 20, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 21, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 22, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 23, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 24, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 25, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 26, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 27, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 28, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 29, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 30, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 31, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 32, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 33, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 34, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 35, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 36, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 37, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 38, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}]
- temporada=2024 atleta_id=72391: [{'rodada': 1, 'clube_id': 266, 'clube_nome': 'Fluminense'}, {'rodada': 2, 'clube_id': 266, 'clube_nome': 'Fluminense'}, {'rodada': 3, 'clube_id': 266, 'clube_nome': 'Fluminense'}, {'rodada': 4, 'clube_id': 266, 'clube_nome': 'Fluminense'}, {'rodada': 5, 'clube_id': 266, 'clube_nome': 'Fluminense'}, {'rodada': 6, 'clube_id': 266, 'clube_nome': 'Fluminense'}, {'rodada': 7, 'clube_id': 266, 'clube_nome': 'Fluminense'}, {'rodada': 8, 'clube_id': 266, 'clube_nome': 'Fluminense'}, {'rodada': 9, 'clube_id': 266, 'clube_nome': 'Fluminense'}, {'rodada': 10, 'clube_id': 266, 'clube_nome': 'Fluminense'}, {'rodada': 27, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 28, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 29, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 30, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 31, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 32, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 33, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 34, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 35, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 36, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 37, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 38, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}]
- temporada=2024 atleta_id=78605: [{'rodada': 1, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 2, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 3, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 4, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 5, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 6, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 7, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 8, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 9, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 10, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 11, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 12, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 13, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 14, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 15, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 16, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 17, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 18, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 19, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 20, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 21, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 22, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 23, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 24, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 25, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 26, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 27, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 28, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 29, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 30, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 31, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 32, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 33, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 34, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 35, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 36, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 37, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 38, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}]
- temporada=2024 atleta_id=79437: [{'rodada': 1, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 2, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 3, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 4, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 5, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 6, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 7, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 8, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 9, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 10, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 11, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 12, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 13, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 14, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 15, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 16, 'clube_id': 286, 'clube_nome': 'Juventude'}, {'rodada': 17, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 18, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 19, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 20, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 21, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 22, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 23, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 24, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 25, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 26, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 27, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 28, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 29, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 30, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 31, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 32, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 33, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 34, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 35, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 36, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 37, 'clube_id': 285, 'clube_nome': 'Internacional'}, {'rodada': 38, 'clube_id': 285, 'clube_nome': 'Internacional'}]
- temporada=2024 atleta_id=83786: [{'rodada': 1, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 2, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 3, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 4, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 5, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 6, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 7, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 8, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 9, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 10, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 11, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 12, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 13, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 14, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 15, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 16, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 17, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 18, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 19, 'clube_id': 283, 'clube_nome': 'Cruzeiro'}, {'rodada': 20, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 21, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 22, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 23, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 24, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 25, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 26, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 27, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 28, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 29, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 30, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 31, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 32, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 33, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 34, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 35, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 36, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 37, 'clube_id': 287, 'clube_nome': 'Vitória'}, {'rodada': 38, 'clube_id': 287, 'clube_nome': 'Vitória'}]
- temporada=2024 atleta_id=84263: [{'rodada': 1, 'clube_id': 282, 'clube_nome': 'Atlético-MG'}, {'rodada': 2, 'clube_id': 282, 'clube_nome': 'Atlético-MG'}, {'rodada': 3, 'clube_id': 282, 'clube_nome': 'Atlético-MG'}, {'rodada': 4, 'clube_id': 282, 'clube_nome': 'Atlético-MG'}, {'rodada': 5, 'clube_id': 282, 'clube_nome': 'Atlético-MG'}, {'rodada': 6, 'clube_id': 282, 'clube_nome': 'Atlético-MG'}, {'rodada': 7, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 8, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 9, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 10, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 11, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 12, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 13, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 14, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 15, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 16, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 17, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 18, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 19, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 20, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 21, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 22, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 23, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 24, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 25, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 26, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 27, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 28, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 29, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 30, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 31, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 32, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 33, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 34, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 35, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 36, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 37, 'clube_id': 284, 'clube_nome': 'Grêmio'}, {'rodada': 38, 'clube_id': 284, 'clube_nome': 'Grêmio'}]
- temporada=2024 atleta_id=85465: [{'rodada': 1, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 2, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 3, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 4, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 5, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 6, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 7, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 8, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 9, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 10, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 11, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 12, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 13, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 14, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 15, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 16, 'clube_id': 293, 'clube_nome': 'Athlético-PR'}, {'rodada': 17, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 18, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 19, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 20, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 21, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 22, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 23, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 24, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 25, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 26, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 27, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 28, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 29, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 30, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 31, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 32, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 33, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 34, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 35, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 36, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 37, 'clube_id': 264, 'clube_nome': 'Corinthians'}, {'rodada': 38, 'clube_id': 264, 'clube_nome': 'Corinthians'}]

## 3.7 Identidade e posicao

- n_pares_temporada_atleta: 2897
- n_pares_com_posicao_variavel_na_mesma_temporada: 1
- n_atletas_com_posicao_distinta_entre_temporadas: 25
- n_clubes_com_multiplos_nomes: 17

Clubes com multiplos nomes (ate 10):

- clube_id=262: ['FLA', 'Flamengo']
- clube_id=263: ['BOT', 'Botafogo']
- clube_id=264: ['COR', 'Corinthians']
- clube_id=265: ['BAH', 'Bahia']
- clube_id=266: ['FLU', 'Fluminense']
- clube_id=267: ['VAS', 'Vasco']
- clube_id=275: ['PAL', 'Palmeiras']
- clube_id=276: ['SAO', 'São Paulo']
- clube_id=280: ['Bragantino', 'RBB']
- clube_id=282: ['Atlético-MG', 'CAM']

## 3.8 Evidencias de elegibilidade (descritivas)

Ver `reports/auditoria_evidencias.csv`.

```
 temporada  rodada     classificacao                                              evidencia                                    impacto_potencial
      2025       1 POSSIVEL_SNAPSHOT prop_entrou=0.0000; prop_zeros=1.0000; pts_mean=0.0000 investigar no harness (possivel snapshot pre-rodada)
```

## Artefatos gerados

- `reports/auditoria_base_canonica_v2.md`
- `reports/auditoria_alvo.csv`
- `reports/auditoria_temporal.csv`
- `reports/auditoria_scouts.csv`
- `reports/auditoria_outliers.csv`
- `reports/auditoria_evidencias.csv`

---

## Apêndice — Decisões finais da ETAPA 3

Data: 2026-09-20
Base: data/interim/cartola_jogador_rodada_v2.parquet

### Decisão A — Elegibilidade do alvo

Todas as observações pós-rodada válidas entram, incluindo
`entrou_em_campo=False`. A única exceção é 2025/R1, excluída por ser
snapshot pré-rodada.

- Linhas totais no canônico: 77.253
- Linhas excluídas (2025/R1): 619
- Linhas elegíveis: 76.634

### Decisão B — Linhas ambíguas

108 linhas com `entrou_em_campo=False` e `pontos_num != 0`:

- Permanecem no canônico.
- Serão marcadas com `flag_ambigua_B=True`.
- Excluídas da avaliação principal inicial.
- Elegíveis para análise secundária.

Arquivo de decisões: config/elegibilidade_v2.yaml

### O que a ETAPA 4 deve respeitar

- Features da rodada R usam somente informação < R.
- 2025/R1 nunca aparece em treino nem em teste.
- `flag_ambigua_B=True` fora da métrica principal.
- Walk-forward expansivo, MAE primária, Spearman secundária.
