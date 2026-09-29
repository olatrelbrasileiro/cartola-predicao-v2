from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from scipy.stats import spearmanr

BASE = Path("data/interim/cartola_jogador_rodada_v2.parquet")
F2 = Path("data/interim/exp12/team_context_features.parquet")
OUT = Path("reports/exp12")
OUT.mkdir(parents=True, exist_ok=True)

F2_FEATURES = [
    "team_goals_scored_last3",
    "team_goals_scored_last5",
    "team_goals_conceded_last3",
    "team_goals_conceded_last5",
    "team_clean_sheet_rate_last3",
    "team_clean_sheet_rate_last5",
    "team_shots_last3",
    "team_shots_last5",
    "team_shots_on_target_last3",
    "team_shots_on_target_last5",
    "opp_goals_scored_last3",
    "opp_goals_scored_last5",
    "opp_goals_conceded_last3",
    "opp_goals_conceded_last5",
    "opp_clean_sheet_rate_last3",
    "opp_clean_sheet_rate_last5",
    "opp_shots_last3",
    "opp_shots_last5",
    "opp_shots_on_target_last3",
    "opp_shots_on_target_last5",
]

df = pd.read_parquet(BASE)
f2 = pd.read_parquet(F2)

# Somente 2026
df = df[df["temporada"] == 2026].copy()

# Merge exato jogador x clube x rodada
df = df.merge(
    f2[
        ["temporada", "rodada", "clube_id"] + F2_FEATURES
    ],
    on=["temporada", "rodada", "clube_id"],
    how="left",
    validate="many_to_one",
)

# Só jogadores que entraram em campo
df = df[df["entrou_em_campo"]].copy()

df = df.sort_values(["rodada", "atleta_id"]).reset_index(drop=True)

params = dict(
    loss_function="MAE",
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=42,
    thread_count=-1,
    verbose=False,
    allow_writing_files=False,
)

pred_rows = []
fold_rows = []

rounds = sorted(df["rodada"].unique())

for R in rounds:
    tr = df[df["rodada"] < R].copy()
    te = df[df["rodada"] == R].copy()

    # F2 precisa existir no treino e no teste
    tr = tr[tr[F2_FEATURES].notna().any(axis=1)].copy()
    te = te[te[F2_FEATURES].notna().any(axis=1)].copy()

    if len(tr) == 0 or len(te) == 0:
        continue

    X_tr = tr[F2_FEATURES]
    y_tr = tr["pontos_num"]

    X_te = te[F2_FEATURES]
    y_te = te["pontos_num"]

    model = CatBoostRegressor(**params)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)

    mae = np.mean(np.abs(y_te.to_numpy() - pred))
    rmse = np.sqrt(np.mean((y_te.to_numpy() - pred) ** 2))

    if len(te) > 1 and np.std(pred) > 0 and np.std(y_te) > 0:
        sp = spearmanr(y_te, pred).statistic
    else:
        sp = np.nan

    fold_rows.append({
        "temporada": 2026,
        "rodada": R,
        "n_train": len(tr),
        "n_eval": len(te),
        "mae": mae,
        "rmse": rmse,
        "spearman": sp,
    })

    pred_rows.append(pd.DataFrame({
        "temporada": 2026,
        "rodada": R,
        "atleta_id": te["atleta_id"].to_numpy(),
        "clube_id": te["clube_id"].to_numpy(),
        "posicao_id": te["posicao_id"].to_numpy(),
        "y_true": y_te.to_numpy(),
        "y_pred": pred,
    }))

folds = pd.DataFrame(fold_rows)
preds = pd.concat(pred_rows, ignore_index=True)

mae = np.mean(np.abs(preds["y_true"] - preds["y_pred"]))
rmse = np.sqrt(np.mean((preds["y_true"] - preds["y_pred"]) ** 2))
sp = spearmanr(preds["y_true"], preds["y_pred"]).statistic

print()
print("=== EXP-12 F2 ONLY — 2026 ===")
print(f"features: {len(F2_FEATURES)}")
print(f"folds: {len(folds)}")
print(f"predicoes: {len(preds)}")
print()
print(f"MAE      = {mae:.4f}")
print(f"RMSE     = {rmse:.4f}")
print(f"Spearman = {sp:.4f}")
print()

print("=== POR POSICAO ===")
for pos, g in preds.groupby("posicao_id"):
    pmae = np.mean(np.abs(g["y_true"] - g["y_pred"]))
    prmse = np.sqrt(np.mean((g["y_true"] - g["y_pred"]) ** 2))
    psp = spearmanr(g["y_true"], g["y_pred"]).statistic
    print(
        f"posicao {pos}: n={len(g)} "
        f"MAE={pmae:.4f} RMSE={prmse:.4f} SP={psp:.4f}"
    )

print()
print("=== POR RODADA ===")
print(
    folds[
        ["rodada", "n_train", "n_eval", "mae", "rmse", "spearman"]
    ].to_string(index=False)
)

preds.to_csv(OUT / "catboost_f2_only_2026_predictions.csv", index=False)
folds.to_csv(OUT / "catboost_f2_only_2026_folds.csv", index=False)

print()
print("Arquivos salvos em reports/exp12/")
