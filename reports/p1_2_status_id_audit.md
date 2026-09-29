# P1.2 — Auditoria temporal do status_id

- Gerado em: 2026-09-21T12:55:40
- Classificacao padrao: **INDETERMINADO**

## AVISO METODOLOGICO

- Correlacao entre status_id e entrou_em_campo NAO prova temporalidade.
- Persistencia entre rodadas NAO prova temporalidade.
- Sem documentacao oficial ou timestamp de coleta, classificacao = INDETERMINADO.

## 1. Distribuicao

| status_id | n | % |
|---|---|---|
| 2 | 1964 | 2.54% |
| 3 | 1328 | 1.72% |
| 5 | 5246 | 6.79% |
| 6 | 46750 | 60.52% |
| 7 | 21965 | 28.43% |

## 2. P(entrou | status_id) — DIAGNOSTICO, NAO PROVA

| status_id | n | n_part | p_entrou |
|---|---|---|---|
| 2 | 1964 | 1345 | 0.6848 |
| 3 | 1328 | 994 | 0.7485 |
| 5 | 5246 | 172 | 0.0328 |
| 6 | 46750 | 10814 | 0.2313 |
| 7 | 21965 | 18490 | 0.8418 |

**AVISO:** associacao contemporanea. Nao informa temporalidade.

## 3. Transicoes entre rodadas consecutivas

- transicoes: 74318
- permanencia: 58120/74318 (78.2%)
- **AVISO:** permanencia alta nao distingue pre-rodada de pos-rodada estavel.

## 4. Documentacao interna

- ocorrencias: 40

  - reports/catalogo_dados_caRtola.md:44  |  | atletas.slug | Jogador | 2024, 2025, 2026 |
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:8  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:19  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:26  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:33  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:40  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:47  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:54  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:61  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:68  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:75  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:82  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:89  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:96  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:103  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:110  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:117  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:124  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:131  |  
  - reports/comparacao_schema_caRtola_2024_2025_2026.md:138  |  

## 5. Caminhos explicitos para PASS

Para transformar INDETERMINADO em PASS, e' necessario UM dos dois:

**(A) Documentacao oficial** que estabeleca que `status_id` representa um estado do jogador disponivel ANTES da rodada (ex.: pagina GE/Cartola, glossario da API oficial).

**(B) Evidencia temporal de coleta** que demonstre, com timestamp, que `status_id` era registrado antes do inicio da rodada (ex.: snapshot de coleta com hora anterior ao cutoff).

Sem A ou B, a classificacao permanece INDETERMINADO.

## 6. Classificacao final

**FATO:** status_id correlaciona com entrou_em_campo; permanencia alta.
**INFERENCIA:** padrao compativel com status de mercado pre-rodada.
**HIPOTESE:** status_id pode codificar estado pre-rodada. Nao testada.
**RECOMENDACAO:** nao usar como feature. Reabrir se A ou B for obtido.

## Status final: **INDETERMINADO**

## Integridade

- Parquet inalterado: True
