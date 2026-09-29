# Harness walk-forward v2 — relatorio

- **Inicio:** 2026-09-20T06:55:09
- **Fim:** 2026-09-20T06:55:14
- **Duracao:** 5.11s

## 1. Politica carregada

- Arquivo: `config/elegibilidade_v2.yaml`
- min_train_rounds (harness): 9
- Walk-forward expansivo intra-temporada.

## 2. Numero total de linhas

- 77253

## 3. Linhas excluidas (2025/R1)

- 619

## 4. Linhas flag_ambigua_B

- 108
- Permanecem no dataframe; excluidas apenas da metrica principal.

## 5. Numero de folds por temporada

- 2024: 29
- 2025: 29
- 2026: 18

## 6. Numero total de folds

- 76

## 7. Tabela-resumo dos folds

```
 temporada  rodada_alvo  train_min_round  train_max_round  test_round  n_train_eligible  n_test_eligible  n_test_ambiguous_B  n_test_used_for_metrics
      2024           10                1                9          10              6301              717                   1                      716
      2024           11                1               10          11              7018              720                   1                      719
      2024           12                1               11          12              7738              722                   0                      722
      2024           13                1               12          13              8460              724                   1                      723
      2024           14                1               13          14              9184              734                   0                      734
      2024           15                1               14          15              9918              738                   1                      737
      2024           16                1               15          16             10656              739                   3                      736
      2024           17                1               16          17             11395              758                   0                      758
      2024           18                1               17          18             12153              762                   4                      758
      2024           19                1               18          19             12915              765                   2                      763
      2024           20                1               19          20             13680              769                   3                      766
      2024           21                1               20          21             14449              754                   1                      753
      2024           22                1               21          22             15203              753                   0                      753
      2024           23                1               22          23             15956              757                   2                      755
      2024           24                1               23          24             16713              766                   3                      763
      2024           25                1               24          25             17479              779                   0                      779
      2024           26                1               25          26             18258              786                   0                      786
      2024           27                1               26          27             19044              780                   0                      780
      2024           28                1               27          28             19824              783                   3                      780
      2024           29                1               28          29             20607              784                   3                      781
      2024           30                1               29          30             21391              786                   0                      786
      2024           31                1               30          31             22177              787                   2                      785
      2024           32                1               31          32             22964              787                   0                      787
      2024           33                1               32          33             23751              788                   0                      788
      2024           34                1               33          34             24539              790                   1                      789
      2024           35                1               34          35             25329              789                   1                      788
      2024           36                1               35          36             26118              792                   1                      791
      2024           37                1               36          37             26910              793                   2                      791
      2024           38                1               37          38             27703              796                   3                      793
      2025           10                2                9          10              5616              720                   0                      720
      2025           11                2               10          11              6336              726                   0                      726
      2025           12                2               11          12              7062              727                   1                      726
      2025           13                2               12          13              7789              718                   0                      718
      2025           14                2               13          14              8507              724                   0                      724
      2025           15                2               14          15              9231              730                   0                      730
      2025           16                2               15          16              9961              732                   1                      731
      2025           17                2               16          17             10693              740                   2                      738
      2025           18                2               17          18             11433              742                   1                      741
      2025           19                2               18          19             12175              747                   2                      745
      2025           20                2               19          20             12922              756                   1                      755
      2025           21                2               20          21             13678              767                   1                      766
      2025           22                2               21          22             14445              774                   1                      773
      2025           23                2               22          23             15219              774                   2                      772
      2025           24                2               23          24             15993              775                   0                      775
      2025           25                2               24          25             16768              786                   1                      785
      2025           26                2               25          26             17554              787                   0                      787
      2025           27                2               26          27             18341              789                   2                      787
      2025           28                2               27          28             19130              797                   1                      796
      2025           29                2               28          29             19927              798                   0                      798
      2025           30                2               29          30             20725              800                   0                      800
      2025           31                2               30          31             21525              802                   1                      801
      2025           32                2               31          32             22327              801                   0                      801
      2025           33                2               32          33             23128              802                   0                      802
      2025           34                2               33          34             23930              803                   1                      802
      2025           35                2               34          35             24733              805                   0                      805
      2025           36                2               35          36             25538              806                   1                      805
      2025           37                2               36          37             26344              807                   0                      807
      2025           38                2               37          38             27151              819                   2                      817
      2026           10                1                9          10              6451              740                   2                      738
      2026           11                1               10          11              7191              743                   0                      743
      2026           12                1               11          12              7934              745                   2                      743
      2026           13                1               12          13              8679              747                   3                      744
      2026           14                1               13          14              9426              750                   1                      749
      2026           15                1               14          15             10176              750                   1                      749
      2026           16                1               15          16             10926              754                   1                      753
      2026           17                1               16          17             11680              761                   1                      760
      2026           18                1               17          18             12441              774                   1                      773
      2026           19                1               18          19             13215              771                   0                      771
      2026           20                1               19          20             13986              773                   2                      771
      2026           21                1               20          21             14759              767                   0                      767
      2026           22                1               21          22             15526              765                   2                      763
      2026           23                1               22          23             16291              770                   2                      768
      2026           24                1               23          24             17061              773                   0                      773
      2026           25                1               24          25             17834              775                   3                      772
      2026           26                1               25          26             18609              775                   3                      772
      2026           27                1               26          27             19384              781                   1                      780
```

## 8. Testes de leakage

- TESTE 1 (train_max < rodada_alvo): PASS
- TESTE 2 (test_round == rodada_alvo): PASS
- TESTE 3 (sem interseccao treino/teste): PASS
- TESTE 4 (2025/R1 fora do treino): PASS
- TESTE 5 (2025/R1 fora do teste): PASS
- TESTE 6 (sem B na metrica principal): PASS
- TESTE 7 (contagem de folds): PASS
- TESTE 8 (sem mistura de temporadas): PASS

## 9. Imutabilidade do Parquet

- SHA-256 (antes): `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`
- SHA-256 (depois): `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`
- Linhas antes/depois: 77253/77253
- Colunas iguais: True
- **Imutavel: True**

## 10. Conclusao

Harness pronto para receber baselines (B0–B5) e modelos (Ridge, CatBoost).
Nenhuma feature criada nesta etapa. Nenhum modelo treinado.
