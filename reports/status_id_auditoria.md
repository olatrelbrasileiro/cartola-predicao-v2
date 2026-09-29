# Auditoria do `status_id`

- Gerado em: 2026-09-21T20:25:40
- Parquet: `data/interim/cartola_jogador_rodada_v2.parquet`
- SHA-256: `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`
- Linhas: 77253
- Duplicatas `(temporada, rodada, atleta_id)`: 0
- Celulas `(T,R,A)` com mais de um `status_id`: 0

## 0. Linhagem do campo `status_id`

- Scripts que referenciam o Parquet canonico: ['scripts/aplicar_mascara_temporal_v1.py', 'scripts/p1_3_numeric_2025_audit.py', 'scripts/testar_b1.py', 'scripts/match_odds_cartola.py', 'scripts/testar_b0.py', 'scripts/harness_walk_forward_v2.py', 'scripts/auditar_features_catboost_v1.py', 'scripts/testar_catboost_v1.py', 'scripts/testar_auditoria_b2_b3.py', 'scripts/baseline_b2s.py', 'scripts/catboost_v1.py', 'scripts/testar_harness_v2.py', 'scripts/p1_1_k7_justification.py', 'scripts/ridge_v1.py', 'scripts/testar_catboost_v2.py', 'scripts/testar_ridge_v2_ablation.py', 'scripts/auditar_base_canonica_v2.py', 'scripts/auditar_spearman_catboost_v1.py', 'scripts/build_historico_shrinkage_v1.py', 'scripts/auditar_etapa5_consistencia.py', 'scripts/normalizar_caRtola_v2.py', 'scripts/etapa_b_k_study.py', 'scripts/testar_ridge_v1.py', 'scripts/baseline_b2.py', 'scripts/testar_ridge_v2.py', 'scripts/auditoria_h2a_alpha_walkforward.py', 'scripts/auditar_status_id.py', 'scripts/baselines_p0_corrected.py', 'scripts/testar_auditoria_h1.py', 'scripts/baseline_b3.py', 'scripts/catboost_v2.py', 'scripts/testar_b2s.py', 'scripts/testar_auditoria_h2.py', 'scripts/baseline_b0.py', 'scripts/build_features_confronto_v1.py', 'scripts/baseline_b1.py', 'scripts/build_ga_shrinkage_v1.py', 'scripts/build_vinculo_atleta_partida.py', 'scripts/testar_b2.py', 'scripts/testar_b3.py', 'scripts/testar_p0_correction.py', 'scripts/p1_2_status_id_audit.py', 'scripts/auditoria_h1_b3_compressao.py', 'scripts/testar_shrinkage_v1.py', 'scripts/auditar_b2_b3.py', 'scripts/testar_confronto_v1.py', 'scripts/ridge_v2.py', 'scripts/ridge_v2_ablation.py']
- Script com mapeamento `atletas.status_id` -> `status_id`: `scripts/p1_3_numeric_2025_audit.py`

Mapeamento confirmado por leitura do script de normalizacao:

```
RAW (data/raw/caRtola/<ano>/rodada-N.csv)
    atletas.status_id        -> status_id
    atletas.entrou_em_campo  -> entrou_em_campo
    atletas.pontos_num       -> pontos_num
```

## 1. Codigos encontrados

| status_id | count | pct |
|---|---|---|
| 2.0 | 1964 | 2.5423% |
| 3.0 | 1328 | 1.719% |
| 5.0 | 5246 | 6.7907% |
| 6.0 | 46750 | 60.5154% |
| 7.0 | 21965 | 28.4326% |

**Nota:** nenhum significado foi atribuido aos codigos.

## 2. Status x entrou_em_campo (MESMA rodada R)

| status_id | total | entrou | nao entrou | % entrou |
|---|---|---|---|---|
| 7 | 21965 | 18490 | 3475 | 84.1794% |
| 3 | 1328 | 994 | 334 | 74.8494% |
| 2 | 1964 | 1345 | 619 | 68.4827% |
| 6 | 46750 | 10814 | 35936 | 23.1316% |
| 5 | 5246 | 172 | 5074 | 3.2787% |

## 3. Por temporada

| temporada | status_id | total | entrou | nao entrou | % entrou |
|---|---|---|---|---|---|
| 2024 | 7 | 8029 | 6813 | 1216 | 84.8549% |
| 2024 | 3 | 511 | 397 | 114 | 77.6908% |
| 2024 | 2 | 730 | 492 | 238 | 67.3973% |
| 2024 | 6 | 17598 | 4031 | 13567 | 22.906% |
| 2024 | 5 | 1631 | 65 | 1566 | 3.9853% |
| 2025 | 7 | 8171 | 6851 | 1320 | 83.8453% |
| 2025 | 3 | 497 | 326 | 171 | 65.5936% |
| 2025 | 2 | 614 | 398 | 216 | 64.8208% |
| 2025 | 6 | 17250 | 3868 | 13382 | 22.4232% |
| 2025 | 5 | 2057 | 54 | 2003 | 2.6252% |
| 2026 | 3 | 320 | 271 | 49 | 84.6875% |
| 2026 | 7 | 5765 | 4826 | 939 | 83.7121% |
| 2026 | 2 | 620 | 455 | 165 | 73.3871% |
| 2026 | 6 | 11902 | 2915 | 8987 | 24.4917% |
| 2026 | 5 | 1558 | 53 | 1505 | 3.4018% |

## 4. Casos criticos

- `status_nulo_total`: n = 0
- `status_nulo_e_entrou`: n = 0
- `status_nulo_e_pontos_nao_zero`: n = 0
- `status_5_baixo_pct_entrou_3.2787pct`: n = 172
- `status_6_baixo_pct_entrou_23.1316pct`: n = 10814

### Exemplos de baixo `% entrou` com entrada confirmada

| status_id | temporada | rodada | atleta_id | apelido | pontos_num |
|---|---|---|---|---|---|
| 5 | 2024 | 1 | 61188 | Gilberto | 0.8 |
| 5 | 2024 | 1 | 69063 | James Rodríguez | 0.0 |
| 5 | 2024 | 1 | 104330 | Dudu | 3.6 |
| 5 | 2024 | 1 | 107203 | Max | 1.2 |
| 5 | 2024 | 2 | 91908 | Pavón | 11.0 |

## 5. Status nulo — auditoria dedicada

- Linhas com status nulo: 0 (0.0%)
- Por temporada: {}
- Entraram em campo: 0
- Nao entraram: 0
- % que entrou: None%
- Pontos != 0: 0

## 6. Cobertura e duplicatas

- total_linhas: 77253
- n_com_status: 77253
- n_status_nulo: 0
- pct_com_status: 100.0
- n_jogadores_afetados_por_nulo: 0
- n_rodadas_afetadas_por_nulo: 0
- duplicatas_T_R_A: 0
- celulas_T_R_A_com_mais_de_um_status: 0

## 7. RAW x Parquet — comparacao exaustiva

- Rodadas comparadas: 103
- Linhas RAW (total): 77253
- Linhas Parquet (total): 77253
- Interseccao por `(temporada, rodada, atleta_id)`: 77253
- Linhas so no RAW: 0
- Linhas so no Parquet: 0

### Divergencias de valor (linhas em ambos)

- `status_id`: 0
- `entrou_em_campo`: 0
- `pontos_num`: 0

### Por temporada

| temporada | div_status | div_entrou | div_pontos | raw_only | canon_only |
|---|---|---|---|---|---|
| 2024 | 0 | 0 | 0 | 0 | 0 |
| 2025 | 0 | 0 | 0 | 0 | 0 |
| 2026 | 0 | 0 | 0 | 0 | 0 |

### Rodadas com maiores divergencias (top 20)

| temporada | rodada | div_status | div_entrou | div_pontos |
|---|---|---|---|---|
| 2024 | 1 | 0 | 0 | 0 |
| 2024 | 2 | 0 | 0 | 0 |
| 2024 | 3 | 0 | 0 | 0 |
| 2024 | 4 | 0 | 0 | 0 |
| 2024 | 5 | 0 | 0 | 0 |
| 2024 | 6 | 0 | 0 | 0 |
| 2024 | 7 | 0 | 0 | 0 |
| 2024 | 8 | 0 | 0 | 0 |
| 2024 | 9 | 0 | 0 | 0 |
| 2024 | 10 | 0 | 0 | 0 |
| 2024 | 11 | 0 | 0 | 0 |
| 2024 | 12 | 0 | 0 | 0 |
| 2024 | 13 | 0 | 0 | 0 |
| 2024 | 14 | 0 | 0 | 0 |
| 2024 | 15 | 0 | 0 | 0 |
| 2024 | 16 | 0 | 0 | 0 |
| 2024 | 17 | 0 | 0 | 0 |
| 2024 | 18 | 0 | 0 | 0 |
| 2024 | 19 | 0 | 0 | 0 |
| 2024 | 20 | 0 | 0 | 0 |

## 8. Documentacao existente no projeto

- Ocorrencias: 60

  - `ETAPAS.md:72` — tiva quantitativa do k=7 - [ ] P1.2 — Auditoria temporal do status_id - [ ] P1.3 — Auditoria numerica de 2025 - [ ] P1.4 — Atualizacao deste
  - `config/ridge_v2_spec.yaml:89` — csv   # usado por L10  fora_de_escopo:   - "Confronto"   - "status_id"   - "NDCG"   - "xG / xA"   - "embeddings"   - "pooling entre temporad
  - `reports/catalogo_dados_caRtola.md:44` — 6 | | atletas.slug | Jogador | 2024, 2025, 2026 | | atletas.status_id | Jogador | 2024, 2025, 2026 | | atletas.variacao_num | Jogador | 2024
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:8` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['A', 'CA', 'CV', 'DE', '
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:19` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:26` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:33` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:40` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:47` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:54` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:61` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:68` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:75` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:82` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:89` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['PP', 'Unnamed: 0', 'atl
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:96` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:103` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:110` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:117` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m
  - `reports/comparacao_schema_caRtola_2024_2025_2026.md:124` — s.preco_num', 'atletas.rodada_id', 'atletas.slug', 'atletas.status_id', 'atletas.variacao_num'] - Somente em 2024: ['Unnamed: 0', 'atletas.m

## 9. Integridade

- Parquet SHA-256 (antes): `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`
- Parquet SHA-256 (depois): `15be1a317afb940c4ffc0ad0758a034ef0a8438637d58b75265620b47db88db4`
- Inalterado: **True**

---

## RECOMENDACAO PARA ELEGIBILIDADE

Esta secao resume APENAS o que os dados demonstram.
Nenhum filtro de exclusao e' definido nesta etapa.

**FATOS OBSERVADOS:**

- Codigos de `status_id` presentes: `[2, 3, 5, 6, 7]`
- Status nulos: 0 linhas (0.0%)
- Duplicatas `(T, R, A)`: 0
- Celulas `(T,R,A)` com mais de um status: 0
- RAW x Parquet: 77253 linhas em comum; div_status=0, div_entrou=0, div_pontos=0.

**Ranking empirico (por % que entrou na MESMA rodada R):**

- `status_id = 7`: 84.1794% entraram (18490/21965)
- `status_id = 3`: 74.8494% entraram (994/1328)
- `status_id = 2`: 68.4827% entraram (1345/1964)
- `status_id = 6`: 23.1316% entraram (10814/46750)
- `status_id = 5`: 3.2787% entraram (172/5246)

**Pendencias antes de qualquer decisao de elegibilidade:**

- Nenhum codigo e' rotulado como duvida/provavel/contundido aqui. A atribuicao de significado deve vir de documentacao/proveniencia.
- Se algum codigo combina baixo `% entrou` com entradas confirmadas, a causa deve ser investigada antes de virar filtro de exclusao.
- Se status nulo tem entradas confirmadas em volume, tratar nulo como nao elegivel precisa de justificativa adicional.
