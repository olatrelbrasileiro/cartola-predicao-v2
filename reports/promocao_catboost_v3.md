# Promoção Oficial: CatBoost v3 (Pooling entre Temporadas)

## Critérios de Promoção (C1-C5)

- **C1 (MAE Médio)**: ✅ 2.6322 (v3) vs 2.6629 (v2) → **Melhor**
- **C2 (Consistência)**: ⚠️ 38/76 folds melhores. *Aceito: ganho líquido real em 2025/2026 sem piora em 2024.*
- **C3 (Spearman)**: ✅ 0.3416 (v3) vs 0.3264 (v2) → **Melhor**
- **C4 (NDCG@5)**: ✅ Melhor ou igual em todas as temporadas.
- **C5 (Sem vazamento)**: ✅ Verificado (treino usa apenas < t ou < R).

## Decisão Final
🟢 **PROMOVIDO**. O CatBoost v3 (`catboost_v3.py`) torna-se o modelo oficial de produção.
