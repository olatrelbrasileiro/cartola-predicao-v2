#!/usr/bin/env python
"""
scripts/exp12_gol_00_audit_vs_official.py

AUDITORIA NÃO DESTRUTIVA — EXP-12 GOL

Objetivo
--------
Descobrir em qual etapa o sinal preditivo dos goleiros pode estar sendo
perdido em relação ao pipeline CatBoost usado como referência no projeto.

A cadeia auditada é:

    CSV RAW
       ↓
    cartola_jogador_rodada_exp12
       ↓
    Script 01 / build_individual_features
       ↓
    gol_pooling_2024_2026
       ↓
    A_full vs A_sem_DE

O script NÃO altera datasets, scripts ou pipeline oficial.

Princípios
----------
1. Scout bruto é auditado diretamente na fonte RAW.
2. Acumulado vs delta é tratado como questão empírica.
3. Primeira aparição é evidência, não diagnóstico automático.
4. A transformação real do Script 01 é executada via importlib quando possível.
5. A reconstrução independente é mantida separada da reconstrução do Script 01.
6. gk_DE_mean_3/5 armazenados são comparados contra a transformação real.
7. A_full vs A_sem_DE usa os mesmos folds, linhas e target.
8. Métricas são apresentadas tanto pooled quanto média dos folds.
9. Pipeline oficial só é chamado de "oficial" se identificado explicitamente.
10. O relatório separa observação, evidência, hipótese e conclusão.

Saídas
-------
reports/exp12/
    exp12_gol_audit_vs_official.md
    exp12_gol_target_comparison.csv
    exp12_gol_sample_comparison.csv
    exp12_gol_de_raw_audit.csv
    exp12_gol_de_first_appearance.csv
    exp12_gol_de_raw_vs_exp12.csv
    exp12_gol_feature_comparison.csv
    exp12_gol_controlled_models.csv
    exp12_gol_official_candidates.csv
    exp12_gol_audit_manifest.json
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent

REPORTS = ROOT / "reports" / "exp12"
REPORTS.mkdir(parents=True, exist_ok=True)

JOG_EXP12 = ROOT / "data" / "interim" / "cartola_jogador_rodada_exp12.parquet"
JOG_V2 = ROOT / "data" / "interim" / "cartola_jogador_rodada_v2.parquet"
GOL_POOL = ROOT / "data" / "interim" / "exp12" / "gol_pooling_2024_2026.parquet"

SCRIPT01 = ROOT / "scripts" / "exp12_gol_01_build_features.py"

RAW_ROOT = ROOT / "data" / "raw" / "caRtola"

# Script RAW antigo de auditoria, se existir.
RAW_AUDIT_REFERENCE = ROOT / "scripts" / "exp_gk_00_auditar_scouts_raw.py"

GOLEIRO_POSICAO_ID = 1

SEASONS = [2024, 2025, 2026]

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 4,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "loss_function": "RMSE",
}


# ============================================================================
# FEATURES CONTROLADAS
# ============================================================================

A_FULL = [
    "gk_pontos_last1",
    "gk_pontos_mean_3",
    "gk_pontos_mean_3_n_obs",
    "gk_pontos_mean_5",
    "gk_pontos_mean_5_n_obs",
    "gk_pontos_mean_hist",
    "gk_DE_mean_3",
    "gk_DE_mean_3_n_obs",
    "gk_DE_mean_5",
    "gk_DE_mean_5_n_obs",
    "mando_C",
]

A_SEM_DE = [
    "gk_pontos_last1",
    "gk_pontos_mean_3",
    "gk_pontos_mean_3_n_obs",
    "gk_pontos_mean_5",
    "gk_pontos_mean_5_n_obs",
    "gk_pontos_mean_hist",
    "mando_C",
]


# ============================================================================
# SAÍDAS
# ============================================================================

OUT_MD = REPORTS / "exp12_gol_audit_vs_official.md"

OUT_TARGET = REPORTS / "exp12_gol_target_comparison.csv"
OUT_SAMPLE = REPORTS / "exp12_gol_sample_comparison.csv"

OUT_DE_RAW = REPORTS / "exp12_gol_de_raw_audit.csv"
OUT_DE_FIRST = REPORTS / "exp12_gol_de_first_appearance.csv"
OUT_DE_RAW_VS = REPORTS / "exp12_gol_de_raw_vs_exp12.csv"

OUT_FEATURES = REPORTS / "exp12_gol_feature_comparison.csv"
OUT_CONTROLLED = REPORTS / "exp12_gol_controlled_models.csv"

OUT_OFFICIAL = REPORTS / "exp12_gol_official_candidates.csv"

OUT_MANIFEST = REPORTS / "exp12_gol_audit_manifest.json"


# ============================================================================
# UTILITÁRIOS
# ============================================================================

def sha256_file(path: Path) -> str:
    if not path.exists():
        return "MISSING"

    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def spearman(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    mask = np.isfinite(y) & np.isfinite(p)

    y = y[mask]
    p = p[mask]

    if len(y) < 3:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    if len(np.unique(p)) < 2:
        return np.nan

    try:
        return float(spearmanr(y, p).statistic)
    except Exception:
        return np.nan


def pearson(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    mask = np.isfinite(y) & np.isfinite(p)

    y = y[mask]
    p = p[mask]

    if len(y) < 3:
        return np.nan

    if np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return np.nan

    try:
        return float(pearsonr(y, p).statistic)
    except Exception:
        return np.nan


def stats(series) -> dict:
    v = pd.to_numeric(pd.Series(series), errors="coerce").astype(float)

    nn = v.dropna()

    if len(nn) == 0:
        return {
            "n": int(len(v)),
            "nan": int(v.isna().sum()),
            "n_neg": 0,
            "n_zero": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "median": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    return {
        "n": int(len(v)),
        "nan": int(v.isna().sum()),
        "n_neg": int((nn < 0).sum()),
        "n_zero": int((nn == 0).sum()),
        "mean": float(nn.mean()),
        "std": float(nn.std()),
        "min": float(nn.min()),
        "p01": float(nn.quantile(0.01)),
        "median": float(nn.median()),
        "p99": float(nn.quantile(0.99)),
        "max": float(nn.max()),
    }


def md_table(df: pd.DataFrame | None) -> str:
    if df is None or len(df) == 0:
        return "_(vazio)_"

    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


# ============================================================================
# FASE 0 — SCRIPT 01
# ============================================================================

def load_script01():
    """
    Importa o Script 01 sem executar main().
    """

    if not SCRIPT01.exists():
        return None, f"Script 01 não encontrado: {SCRIPT01}"

    try:
        spec = importlib.util.spec_from_file_location(
            "exp12_gol_build_features",
            str(SCRIPT01),
        )

        if spec is None or spec.loader is None:
            return None, "importlib não conseguiu criar o loader."

        module = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(module)

        return module, "OK"

    except Exception as exc:
        return None, (
            f"Falha ao importar Script 01: "
            f"{type(exc).__name__}: {exc}"
        )


def script01_source_excerpt() -> str:
    """
    Extrai trechos relevantes do Script 01 para o relatório.
    """

    if not SCRIPT01.exists():
        return "(Script 01 não encontrado.)"

    text = SCRIPT01.read_text(errors="ignore")

    patterns = [
        r"def _to_delta\(.*?(?=\ndef |\Z)",
        r"def build_individual_features\(.*?(?=\ndef |\Z)",
        r"ACCUMULATED_TEMPORADAS\s*=.*",
    ]

    output = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.DOTALL,
        )

        for match in matches[:2]:

            output.append("```python")
            output.append(match.rstrip())
            output.append("```\n")

    return "\n".join(output) if output else "(trechos não encontrados)"


# ============================================================================
# FASE 1 — DESCOBERTA DO PIPELINE OFICIAL
# ============================================================================

KEYWORDS = [
    "CatBoost",
    "catboost",
    "CatBoostRegressor",
    "posicao_id",
    "walk_forward",
    "walkforward",
    "goleiro",
    "GOL",
    "pontos_num",
]


def discover_official_candidates() -> pd.DataFrame:

    scripts_dir = ROOT / "scripts"

    if not scripts_dir.exists():
        return pd.DataFrame()

    rows = []

    for path in sorted(scripts_dir.rglob("*.py")):

        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue

        hits = {
            kw: len(re.findall(re.escape(kw), text))
            for kw in KEYWORDS
        }

        total_hits = sum(hits.values())

        if total_hits == 0:
            continue

        rows.append({
            "script": str(path.relative_to(ROOT)),
            "hits_total": total_hits,
            "hits_catboost": hits["CatBoost"] + hits["catboost"],
            "hits_catboost_regressor": hits["CatBoostRegressor"],
            "hits_posicao_id": hits["posicao_id"],
            "hits_walkforward": hits["walk_forward"] + hits["walkforward"],
            "hits_goleiro": hits["goleiro"] + hits["GOL"],
            "hits_pontos_num": hits["pontos_num"],
            "has_fit": (
                "CatBoostRegressor(" in text
                or ".fit(" in text
            ),
            "has_posicao_1": bool(
                re.search(
                    r"posicao_id\s*==\s*1",
                    text,
                )
            ),
            "size_bytes": path.stat().st_size,
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["hits_catboost_regressor", "hits_total"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================================
# FASE 2 — CARREGAMENTO DO RAW CSV
# ============================================================================

def find_raw_files() -> list[tuple[int, int, Path]]:

    files = []

    for season in SEASONS:

        season_dir = RAW_ROOT / str(season)

        if not season_dir.exists():
            continue

        for path in sorted(season_dir.glob("rodada-*.csv")):

            match = re.search(
                r"rodada-(\d+)",
                path.stem,
            )

            if not match:
                continue

            rodada = int(match.group(1))

            files.append(
                (season, rodada, path)
            )

    return files


def load_raw_gol_csv() -> tuple[pd.DataFrame, pd.DataFrame]:

    observation_rows = []
    schema_rows = []

    files = find_raw_files()

    if not files:
        return pd.DataFrame(), pd.DataFrame()

    for season, rodada, path in files:

        try:
            df = pd.read_csv(
                path,
                low_memory=False,
            )
        except Exception as exc:

            schema_rows.append({
                "temporada": season,
                "rodada": rodada,
                "arquivo": str(path),
                "erro": f"{type(exc).__name__}: {exc}",
            })

            continue

        id_col = "atletas.atleta_id"
        pos_col = "atletas.posicao_id"
        enter_col = "atletas.entrou_em_campo"
        name_col = "atletas.apelido"

        required = [
            id_col,
            pos_col,
            enter_col,
        ]

        if not all(c in df.columns for c in required):
            continue

        gk = df[
            df[pos_col] == GOLEIRO_POSICAO_ID
        ].copy()

        has_de = "DE" in df.columns

        schema_rows.append({
            "temporada": season,
            "rodada": rodada,
            "arquivo": str(path.relative_to(ROOT)),
            "n_goleiros": len(gk),
            "coluna_DE_existe": has_de,
            "n_DE_nonnull": (
                int(gk["DE"].notna().sum())
                if has_de
                else 0
            ),
            "n_entrou_em_campo": int(
                (gk[enter_col] == True).sum()
            ),
        })

        for _, row in gk.iterrows():

            observation_rows.append({
                "temporada": season,
                "rodada": rodada,
                "atleta_id": row[id_col],
                "apelido_raw": (
                    row[name_col]
                    if name_col in df.columns
                    else None
                ),
                "entrou_em_campo_raw": row[enter_col],
                "DE_raw": (
                    safe_float(row["DE"])
                    if has_de
                    else np.nan
                ),
                "DE_coluna_existe": has_de,
                "arquivo_raw": str(
                    path.relative_to(ROOT)
                ),
            })

    raw = pd.DataFrame(observation_rows)

    schema = pd.DataFrame(schema_rows)

    if len(raw):

        raw = raw.sort_values(
            [
                "temporada",
                "atleta_id",
                "rodada",
            ],
            kind="mergesort",
        ).reset_index(drop=True)

    return raw, schema


# ============================================================================
# FASE 3 — NATUREZA DO SCOUT_DE RAW
# ============================================================================

def audit_raw_de(raw: pd.DataFrame) -> pd.DataFrame:

    if raw.empty:
        return pd.DataFrame()

    rows = []

    for season in sorted(raw["temporada"].unique()):

        sub = raw[
            (raw["temporada"] == season)
            & raw["DE_coluna_existe"]
            & raw["DE_raw"].notna()
        ].copy()

        sub = sub.sort_values(
            [
                "atleta_id",
                "rodada",
            ],
            kind="mergesort",
        )

        diffs = (
            sub.groupby(
                "atleta_id",
                sort=False,
            )["DE_raw"]
            .diff()
            .dropna()
        )

        rows.append({
            "temporada": int(season),
            "n_observacoes_raw_DE": int(len(sub)),
            "n_atletas": int(sub["atleta_id"].nunique()),
            "n_pairs_temporais": int(len(diffs)),

            "pct_diff_negativa": (
                float((diffs < 0).mean())
                if len(diffs)
                else np.nan
            ),

            "pct_diff_zero": (
                float((diffs == 0).mean())
                if len(diffs)
                else np.nan
            ),

            "pct_diff_positiva": (
                float((diffs > 0).mean())
                if len(diffs)
                else np.nan
            ),

            "median_raw": float(sub["DE_raw"].median()),
            "p99_raw": float(sub["DE_raw"].quantile(.99)),
            "max_raw": float(sub["DE_raw"].max()),

            "median_diff": (
                float(diffs.median())
                if len(diffs)
                else np.nan
            ),

            "p99_abs_diff": (
                float(diffs.abs().quantile(.99))
                if len(diffs)
                else np.nan
            ),

            "max_abs_diff": (
                float(diffs.abs().max())
                if len(diffs)
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


# ============================================================================
# FASE 4 — PRIMEIRA OBSERVAÇÃO REAL
# ============================================================================

def first_appearance_audit(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    if raw.empty:
        return pd.DataFrame()

    sub = raw[
        raw["DE_coluna_existe"]
        & raw["DE_raw"].notna()
    ].copy()

    sub = sub.sort_values(
        [
            "temporada",
            "atleta_id",
            "rodada",
        ],
        kind="mergesort",
    )

    first = (
        sub.groupby(
            ["temporada", "atleta_id"],
            sort=False,
        )
        .first()
        .reset_index()
    )

    first = first[
        [
            "temporada",
            "atleta_id",
            "rodada",
            "DE_raw",
            "entrou_em_campo_raw",
            "arquivo_raw",
        ]
    ].rename(
        columns={
            "rodada": "primeira_rodada_com_DE",
            "DE_raw": "primeiro_DE_raw",
        }
    )

    first["bucket"] = np.where(
        first["primeira_rodada_com_DE"] == 1,
        "R1",
        "R>1",
    )

    rows = []

    for (season, bucket), g in first.groupby(
        ["temporada", "bucket"],
        sort=False,
    ):

        rows.append({
            "temporada": int(season),
            "bucket": bucket,
            "n_goleiros": int(len(g)),
            "median_primeiro_DE": float(
                g["primeiro_DE_raw"].median()
            ),
            "p99_primeiro_DE": float(
                g["primeiro_DE_raw"].quantile(.99)
            ),
            "max_primeiro_DE": float(
                g["primeiro_DE_raw"].max()
            ),
            "n_gt_15": int(
                (g["primeiro_DE_raw"] > 15).sum()
            ),
            "n_gt_30": int(
                (g["primeiro_DE_raw"] > 30).sum()
            ),
            "n_gt_50": int(
                (g["primeiro_DE_raw"] > 50).sum()
            ),
        })

    summary = pd.DataFrame(rows)

    # Acrescenta amostras grandes para inspeção manual.
    large = first[
        first["primeiro_DE_raw"] > 15
    ].copy()

    large = large.sort_values(
        "primeiro_DE_raw",
        ascending=False,
    ).head(50)

    return summary, large


# ============================================================================
# FASE 5 — RECONSTRUÇÃO INDEPENDENTE DO RAW
# ============================================================================

def reconstruct_raw_delta(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstrução independente.

    NÃO afirma que esta é a regra correta do Cartola.

    Regra observacional:
      - primeiro valor conhecido → raw
      - valor seguinte → raw - último valor conhecido
      - NaN → NaN

    Serve para comparar a transformação do Script 01.
    """

    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    df = df.sort_values(
        [
            "temporada",
            "atleta_id",
            "rodada",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    delta = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    first_flag = np.zeros(
        len(df),
        dtype=bool,
    )

    for _, indices in df.groupby(
        ["temporada", "atleta_id"],
        sort=False,
    ).groups.items():

        previous = None

        for idx in indices:

            raw_value = df.at[idx, "DE_raw"]

            if pd.isna(raw_value):
                continue

            raw_value = float(raw_value)

            if previous is None:

                delta[idx] = raw_value
                first_flag[idx] = True

            else:

                delta[idx] = raw_value - previous

            previous = raw_value

    df["DE_delta_independente"] = delta
    df["DE_primeira_observacao"] = first_flag

    return df


# ============================================================================
# FASE 6 — TRANSFORMAÇÃO REAL DO SCRIPT 01
# ============================================================================

def run_real_script01(
    jog_exp12: pd.DataFrame,
    module,
):
    """
    Executa a função real build_individual_features() do Script 01.

    Retorna:
      dataframe transformado
      mensagem
    """

    if module is None:
        return None, "Script 01 não carregado."

    if not hasattr(
        module,
        "build_individual_features",
    ):
        return None, (
            "Script 01 não possui "
            "build_individual_features()."
        )

    try:

        result = module.build_individual_features(
            jog_exp12.copy()
        )

        # Alguns scripts retornam:
        #   df
        # Outros:
        #   df, detection
        if isinstance(result, tuple):

            dfs = [
                x for x in result
                if isinstance(x, pd.DataFrame)
            ]

            if not dfs:
                return None, (
                    "build_individual_features retornou "
                    "tuple sem DataFrame."
                )

            return dfs[0], "OK"

        if isinstance(result, pd.DataFrame):
            return result, "OK"

        return None, (
            "Retorno inesperado de "
            "build_individual_features: "
            f"{type(result).__name__}"
        )

    except Exception as exc:

        return None, (
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================================
# FASE 7 — RAW → PARQUET EXP12
# ============================================================================

def compare_raw_to_exp12(
    raw: pd.DataFrame,
    jog_exp12: pd.DataFrame,
) -> pd.DataFrame:

    if raw.empty or jog_exp12 is None:
        return pd.DataFrame()

    keys = [
        "temporada",
        "rodada",
        "atleta_id",
    ]

    # Aqui usamos apenas a coluna raw do CSV.
    raw_cmp = raw[
        keys
        + [
            "DE_raw",
            "entrou_em_campo_raw",
        ]
    ].copy()

    exp_cols = [
        c for c in [
            "scout_DE",
            "entrou_em_campo",
            "pontos_num",
        ]
        if c in jog_exp12.columns
    ]

    exp_cmp = jog_exp12[
        keys + exp_cols
    ].copy()

    merged = raw_cmp.merge(
        exp_cmp,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if "scout_DE" in merged.columns:

        merged["diff_raw_vs_exp12_scout_DE"] = (
            pd.to_numeric(
                merged["scout_DE"],
                errors="coerce",
            )
            - pd.to_numeric(
                merged["DE_raw"],
                errors="coerce",
            )
        )

        merged["abs_diff_raw_vs_exp12"] = (
            merged["diff_raw_vs_exp12_scout_DE"]
            .abs()
        )

    return merged


# ============================================================================
# FASE 8 — SCRIPT 01 REAL VS GOL_POOL
# ============================================================================

def compare_script01_to_pool(
    script01_df: pd.DataFrame | None,
    gol_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    if script01_df is None:
        return pd.DataFrame(), {
            "status": "script01_nao_executado"
        }

    keys = [
        "temporada",
        "rodada",
        "atleta_id",
    ]

    wanted = [
        "gk_DE_mean_3",
        "gk_DE_mean_5",
        "gk_DE_mean_3_n_obs",
        "gk_DE_mean_5_n_obs",
    ]

    available = [
        c for c in wanted
        if c in script01_df.columns
        and c in gol_pool.columns
    ]

    if not available:
        return pd.DataFrame(), {
            "status": "features_gk_DE_nao_encontradas"
        }

    left = gol_pool[
        keys + available
    ].copy()

    right = script01_df[
        keys + available
    ].copy()

    merged = left.merge(
        right,
        on=keys,
        how="inner",
        suffixes=(
            "_pool",
            "_script01",
        ),
        validate="one_to_one",
    )

    rows = []

    for feature in available:

        if feature.endswith("_n_obs"):
            continue

        a = pd.to_numeric(
            merged[f"{feature}_pool"],
            errors="coerce",
        )

        b = pd.to_numeric(
            merged[f"{feature}_script01"],
            errors="coerce",
        )

        diff = b - a

        mask = diff.notna()

        rows.append({
            "feature": feature,
            "n_comparavel": int(mask.sum()),
            "n_diferente": int(
                (diff.abs() > 1e-9).sum()
            ),
            "mae_diferenca": float(
                diff.abs().mean()
            ) if mask.any() else np.nan,
            "mediana_diferenca": float(
                diff.median()
            ) if mask.any() else np.nan,
            "max_abs_diferenca": float(
                diff.abs().max()
            ) if mask.any() else np.nan,
            "pearson": pearson(
                a[mask],
                b[mask],
            ),
            "spearman": spearman(
                a[mask],
                b[mask],
            ),
        })

        merged[
            f"diff_{feature}"
        ] = diff

    summary = {
        "status": "OK",
        "n_join": int(len(merged)),
        "features": rows,
    }

    return merged, summary


# ============================================================================
# FASE 9 — CORRELAÇÃO FEATURE × TARGET
# ============================================================================

def feature_target_correlation(
    gol_pool: pd.DataFrame,
) -> pd.DataFrame:

    if "elegivel_modelo" not in gol_pool.columns:
        return pd.DataFrame()

    eligible = (
        gol_pool["elegivel_modelo"]
        .fillna(False)
        .astype(bool)
    )

    features = [
        "gk_DE_mean_3",
        "gk_DE_mean_5",
        "gk_pontos_mean_5",
        "gk_pontos_mean_3",
        "gk_pontos_last1",
        "gk_pontos_mean_hist",
    ]

    features = [
        f for f in features
        if f in gol_pool.columns
    ]

    folds = (
        gol_pool.loc[
            eligible,
            ["temporada", "rodada"],
        ]
        .drop_duplicates()
        .sort_values(
            ["temporada", "rodada"]
        )
        .values.tolist()
    )

    rows = []

    for feature in features:

        pooled = gol_pool.loc[
            eligible
            & gol_pool[feature].notna()
        ]

        y = pd.to_numeric(
            pooled["pontos_num"],
            errors="coerce",
        )

        x = pd.to_numeric(
            pooled[feature],
            errors="coerce",
        )

        rows.append({
            "feature": feature,
            "n_pooled": int(
                (x.notna() & y.notna()).sum()
            ),
            "spearman_pooled": spearman(
                y,
                x,
            ),
        })

    result = pd.DataFrame(rows)

    # Média de Spearman por fold.
    for feature in features:

        values = []

        for T, R in folds:

            mask = (
                eligible
                & (gol_pool["temporada"] == T)
                & (gol_pool["rodada"] == R)
            )

            x = gol_pool.loc[
                mask,
                feature,
            ]

            y = gol_pool.loc[
                mask,
                "pontos_num",
            ]

            values.append(
                spearman(y, x)
            )

        valid = [
            v for v in values
            if np.isfinite(v)
        ]

        result.loc[
            result["feature"] == feature,
            "spearman_mean_folds",
        ] = (
            float(np.mean(valid))
            if valid
            else np.nan
        )

    return result


# ============================================================================
# FASE 10 — TESTE CONTROLADO A_full VS A_sem_DE
# ============================================================================

def controlled_models(
    gol_pool: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    from catboost import CatBoostRegressor

    if "elegivel_modelo" not in gol_pool.columns:
        return pd.DataFrame(), pd.DataFrame()

    eligible = (
        gol_pool["elegivel_modelo"]
        .fillna(False)
        .astype(bool)
    )

    folds = (
        gol_pool.loc[
            eligible,
            ["temporada", "rodada"],
        ]
        .drop_duplicates()
        .sort_values(
            ["temporada", "rodada"]
        )
        .values.tolist()
    )

    arms = {
        "A_full": A_FULL,
        "A_sem_DE": A_SEM_DE,
    }

    predictions = []
    fold_rows = []

    for T, R in folds:

        train_mask = (
            (
                gol_pool["temporada"] < T
            )
            |
            (
                (gol_pool["temporada"] == T)
                &
                (gol_pool["rodada"] < R)
            )
        ) & eligible

        test_mask = (
            (gol_pool["temporada"] == T)
            &
            (gol_pool["rodada"] == R)
            &
            eligible
        )

        if train_mask.sum() < 50:
            continue

        if test_mask.sum() < 3:
            continue

        base_pred = gol_pool.loc[
            test_mask,
            [
                "temporada",
                "rodada",
                "atleta_id",
                "pontos_num",
            ],
        ].copy()

        base_pred = base_pred.rename(
            columns={
                "pontos_num": "pontos_real"
            }
        )

        fold_record = {
            "temporada": int(T),
            "rodada": int(R),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }

        for arm_name, features in arms.items():

            missing = [
                f for f in features
                if f not in gol_pool.columns
            ]

            if missing:
                raise RuntimeError(
                    f"{arm_name}: features ausentes: "
                    f"{missing}"
                )

            X_train = gol_pool.loc[
                train_mask,
                features,
            ]

            y_train = pd.to_numeric(
                gol_pool.loc[
                    train_mask,
                    "pontos_num",
                ],
                errors="coerce",
            )

            X_test = gol_pool.loc[
                test_mask,
                features,
            ]

            y_test = pd.to_numeric(
                gol_pool.loc[
                    test_mask,
                    "pontos_num",
                ],
                errors="coerce",
            )

            model = CatBoostRegressor(
                **CATBOOST_PARAMS
            )

            model.fit(
                X_train,
                y_train,
            )

            pred = model.predict(X_test)

            fold_record[
                f"MAE_{arm_name}"
            ] = float(
                np.mean(
                    np.abs(
                        y_test.to_numpy()
                        - pred
                    )
                )
            )

            fold_record[
                f"RMSE_{arm_name}"
            ] = float(
                np.sqrt(
                    np.mean(
                        (
                            y_test.to_numpy()
                            - pred
                        ) ** 2
                    )
                )
            )

            fold_record[
                f"Spearman_{arm_name}"
            ] = spearman(
                y_test.to_numpy(),
                pred,
            )

            base_pred[
                f"pred_{arm_name}"
            ] = pred

        predictions.append(base_pred)
        fold_rows.append(fold_record)

        print(
            f"  ctrl {T} R{R}: "
            f"full={fold_record['MAE_A_full']:.4f} | "
            f"sem_DE={fold_record['MAE_A_sem_DE']:.4f}"
        )

    if not fold_rows:
        return pd.DataFrame(), pd.DataFrame()

    fold_df = pd.DataFrame(fold_rows)

    pred_df = pd.concat(
        predictions,
        ignore_index=True,
    )

    result_rows = []

    y = pred_df["pontos_real"].astype(float)

    for arm_name in arms:

        p = pred_df[
            f"pred_{arm_name}"
        ].astype(float)

        result_rows.append({
            "arm": arm_name,

            "n_folds": int(
                fold_df[
                    f"MAE_{arm_name}"
                ].notna().sum()
            ),

            "n_pooled": int(len(pred_df)),

            "MAE_mean_folds": float(
                fold_df[
                    f"MAE_{arm_name}"
                ].mean()
            ),

            "RMSE_mean_folds": float(
                fold_df[
                    f"RMSE_{arm_name}"
                ].mean()
            ),

            "Spearman_mean_folds": float(
                fold_df[
                    f"Spearman_{arm_name}"
                ].mean()
            ),

            "MAE_pooled": float(
                np.mean(
                    np.abs(
                        y.to_numpy()
                        - p.to_numpy()
                    )
                )
            ),

            "RMSE_pooled": float(
                np.sqrt(
                    np.mean(
                        (
                            y.to_numpy()
                            - p.to_numpy()
                        ) ** 2
                    )
                )
            ),

            "Spearman_pooled": spearman(
                y,
                p,
            ),
        })

    result = pd.DataFrame(result_rows)

    return result, fold_df


# ============================================================================
# FASE 11 — TARGET / AMOSTRA
# ============================================================================

def target_and_sample(
    jog_exp12: pd.DataFrame,
    gol_pool: pd.DataFrame,
):

    target_rows = []
    sample_rows = []

    gk = jog_exp12[
        jog_exp12["posicao_id"]
        == GOLEIRO_POSICAO_ID
    ].copy()

    target_rows.append({
        "pipeline": "jog_exp12_goleiros",
        "n": len(gk),
        **stats(gk["pontos_num"]),
    })

    participating = gk[
        gk["entrou_em_campo"] == True
    ]

    target_rows.append({
        "pipeline": "jog_exp12_goleiros_participantes",
        "n": len(participating),
        **stats(participating["pontos_num"]),
    })

    target_rows.append({
        "pipeline": "gol_pool",
        "n": len(gol_pool),
        **stats(gol_pool["pontos_num"]),
    })

    if "elegivel_modelo" in gol_pool.columns:

        elig = gol_pool[
            gol_pool["elegivel_modelo"]
            .fillna(False)
            .astype(bool)
        ]

        target_rows.append({
            "pipeline": "gol_pool_elegiveis",
            "n": len(elig),
            **stats(elig["pontos_num"]),
        })

    sample_rows.append({
        "pipeline": "jog_exp12_goleiros",
        "n_total": len(gk),
        "n_participantes": len(participating),
        "n_atletas": gk["atleta_id"].nunique(),
        "n_temporadas": gk["temporada"].nunique(),
        "n_rodadas": gk["rodada"].nunique(),
    })

    sample_rows.append({
        "pipeline": "gol_pool",
        "n_total": len(gol_pool),
        "n_participantes": (
            gol_pool["elegivel_modelo"]
            .fillna(False)
            .astype(bool)
            .sum()
            if "elegivel_modelo" in gol_pool.columns
            else np.nan
        ),
        "n_atletas": gol_pool["atleta_id"].nunique(),
        "n_temporadas": gol_pool["temporada"].nunique(),
        "n_rodadas": gol_pool["rodada"].nunique(),
    })

    return (
        pd.DataFrame(target_rows),
        pd.DataFrame(sample_rows),
    )


# ============================================================================
# FASE 12 — DIAGNÓSTICO AUTOMÁTICO
# ============================================================================

def build_diagnosis(
    raw_audit: pd.DataFrame,
    first_audit: pd.DataFrame,
    raw_vs_exp12: pd.DataFrame,
    script01_comparison: pd.DataFrame,
    controlled: pd.DataFrame,
    correlations: pd.DataFrame,
) -> str:

    lines = []

    lines.append("### Observações\n")

    lines.append(
        "- A natureza de `scout_DE` é avaliada diretamente no RAW."
    )

    lines.append(
        "- Primeiras aparições são analisadas sem classificá-las "
        "automaticamente como erro."
    )

    lines.append(
        "- A transformação real do Script 01 é comparada "
        "ao `gol_pool` quando a função pode ser executada."
    )

    lines.append(
        "- O teste A_full vs A_sem_DE usa a mesma divisão "
        "walk-forward."
    )

    lines.append("\n### Evidências\n")

    if not raw_audit.empty:

        for _, row in raw_audit.iterrows():

            lines.append(
                f"- {int(row['temporada'])}: "
                f"{row['pct_diff_negativa']:.4f} "
                f"das diferenças temporais de DE são negativas, "
                f"{row['pct_diff_zero']:.4f} são zero e "
                f"{row['pct_diff_positiva']:.4f} são positivas."
            )

    if (
        raw_vs_exp12 is not None
        and not raw_vs_exp12.empty
        and "abs_diff_raw_vs_exp12" in raw_vs_exp12
    ):

        diff = raw_vs_exp12[
            "abs_diff_raw_vs_exp12"
        ].dropna()

        lines.append(
            f"- RAW → `cartola_jogador_rodada_exp12`: "
            f"{len(diff)} observações comparáveis; "
            f"{int((diff > 1e-9).sum())} apresentam diferença "
            f"no valor de `scout_DE`."
        )

    if (
        script01_comparison is not None
        and not script01_comparison.empty
    ):

        for _, row in script01_comparison.iterrows():

            lines.append(
                f"- `{row['feature']}`: "
                f"{int(row['n_diferente'])} diferenças entre "
                f"Script 01 executado e `gol_pool`; "
                f"MAE da diferença = "
                f"{row['mae_diferenca']:.6f}."
            )

    if (
        controlled is not None
        and not controlled.empty
        and len(controlled) == 2
    ):

        full = controlled[
            controlled["arm"] == "A_full"
        ].iloc[0]

        no_de = controlled[
            controlled["arm"] == "A_sem_DE"
        ].iloc[0]

        lines.append(
            f"- Teste controlado: MAE pooled "
            f"A_full={full['MAE_pooled']:.4f}; "
            f"A_sem_DE={no_de['MAE_pooled']:.4f}."
        )

    lines.append("\n### Hipóteses que permanecem testáveis\n")

    lines.append(
        "1. O `scout_DE` pode possuir semântica diferente "
        "entre temporadas."
    )

    lines.append(
        "2. A primeira observação de um goleiro pode estar "
        "introduzindo valor acumulado como evento."
    )

    lines.append(
        "3. A transformação do Script 01 pode diferir "
        "do `gol_pool` armazenado."
    )

    lines.append(
        "4. Mesmo com transformação correta, DE pode não "
        "adicionar sinal preditivo incremental."
    )

    lines.append(
        "5. A distribuição do target de goleiros pode limitar "
        "a capacidade de qualquer feature individual."
    )

    lines.append("\n### Não confirmado\n")

    lines.append(
        "Nenhuma das hipóteses acima é classificada como bug "
        "sem evidência correspondente nas fases anteriores."
    )

    lines.append("\n### Próximo passo\n")

    lines.append(
        "Se a auditoria encontrar divergência entre RAW → "
        "parquet → Script 01 → gol_pool, corrigir primeiro "
        "a transformação. Só depois repetir o benchmark "
        "CatBoost oficial vs EXP-12."
    )

    return "\n".join(lines)


# ============================================================================
# RELATÓRIO
# ============================================================================

def write_report(
    sections: list[tuple[str, str]],
):

    lines = [
        "# EXP-12 GOL — auditoria vs pipeline oficial",
        "",
        f"_Gerado em {datetime.now().isoformat(timespec='seconds')}_",
        "",
    ]

    for title, body in sections:

        lines.append(
            f"## {title}"
        )

        lines.append("")

        lines.append(body)

        lines.append("")

    OUT_MD.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("EXP-12 GOL — AUDITORIA FINAL")
    print("=" * 80)

    # ----------------------------------------------------------------------
    # 0. Script 01
    # ----------------------------------------------------------------------

    print("\n[0/11] carregando Script 01...")

    mod01, msg01 = load_script01()

    print(f"      {msg01}")

    excerpt = script01_source_excerpt()

    # ----------------------------------------------------------------------
    # 1. Pipeline oficial
    # ----------------------------------------------------------------------

    print("\n[1/11] procurando candidatos ao pipeline oficial...")

    official_candidates = discover_official_candidates()

    if not official_candidates.empty:

        official_candidates.head(20).to_csv(
            OUT_OFFICIAL,
            index=False,
        )

        print(
            official_candidates[
                [
                    "script",
                    "hits_total",
                    "hits_catboost_regressor",
                    "hits_walkforward",
                    "hits_posicao_id",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    # ----------------------------------------------------------------------
    # 2. Datasets
    # ----------------------------------------------------------------------

    print("\n[2/11] carregando datasets...")

    if not JOG_EXP12.exists():
        raise SystemExit(
            f"Dataset ausente: {JOG_EXP12}"
        )

    if not GOL_POOL.exists():
        raise SystemExit(
            f"Dataset ausente: {GOL_POOL}"
        )

    jog_exp12 = pd.read_parquet(
        JOG_EXP12
    )

    gol_pool = pd.read_parquet(
        GOL_POOL
    )

    print(
        f"      jog_exp12: {jog_exp12.shape}"
    )

    print(
        f"      gol_pool : {gol_pool.shape}"
    )

    raw_parquet_gol = jog_exp12[
        jog_exp12["posicao_id"]
        == GOLEIRO_POSICAO_ID
    ].copy()

    # ----------------------------------------------------------------------
    # 3. RAW CSV
    # ----------------------------------------------------------------------

    print("\n[3/11] lendo CSVs RAW...")

    raw_gol, raw_schema = load_raw_gol_csv()

    print(
        f"      raw_gol: {raw_gol.shape}"
    )

    print(
        f"      arquivos/rodadas: {len(raw_schema)}"
    )

    # ----------------------------------------------------------------------
    # 4. Natureza DE
    # ----------------------------------------------------------------------

    print("\n[4/11] auditando scout_DE bruto...")

    raw_de_audit = audit_raw_de(
        raw_gol
    )

    print(
        md_table(raw_de_audit)
    )

    # ----------------------------------------------------------------------
    # 5. Primeira aparição
    # ----------------------------------------------------------------------

    print(
        "\n[5/11] auditando primeira aparição..."
    )

    first_summary, first_large = (
        first_appearance_audit(
            raw_gol
        )
        if not raw_gol.empty
        else (pd.DataFrame(), pd.DataFrame())
    )

    print(
        md_table(first_summary)
    )

    # ----------------------------------------------------------------------
    # 6. Reconstrução independente
    # ----------------------------------------------------------------------

    print(
        "\n[6/11] reconstrução independente do delta..."
    )

    raw_reconstructed = reconstruct_raw_delta(
        raw_gol
    )

    # ----------------------------------------------------------------------
    # 7. RAW → parquet EXP12
    # ----------------------------------------------------------------------

    print(
        "\n[7/11] comparando RAW → cartola_jogador_rodada_exp12..."
    )

    raw_vs_exp12 = compare_raw_to_exp12(
        raw_gol,
        jog_exp12,
    )

    if not raw_vs_exp12.empty:

        raw_vs_exp12.to_csv(
            OUT_DE_RAW_VS,
            index=False,
        )

        diff = raw_vs_exp12[
            "abs_diff_raw_vs_exp12"
        ].dropna()

        print(
            f"      comparáveis: {len(diff)}"
        )

        print(
            f"      diferentes : "
            f"{int((diff > 1e-9).sum())}"
        )

    # ----------------------------------------------------------------------
    # 8. Executa Script 01 real
    # ----------------------------------------------------------------------

    print(
        "\n[8/11] executando build_individual_features real..."
    )

    script01_df, script01_status = run_real_script01(
        jog_exp12,
        mod01,
    )

    print(
        f"      {script01_status}"
    )

    script01_comparison = pd.DataFrame()
    script01_summary = {}

    if script01_df is not None:

        (
            script01_comparison,
            script01_summary,
        ) = compare_script01_to_pool(
            script01_df,
            gol_pool,
        )

        if not script01_comparison.empty:

            print(
                md_table(script01_comparison)
            )

    # ----------------------------------------------------------------------
    # 9. Target / amostra
    # ----------------------------------------------------------------------

    print(
        "\n[9/11] target e amostra..."
    )

    target_df, sample_df = target_and_sample(
        jog_exp12,
        gol_pool,
    )

    target_df.to_csv(
        OUT_TARGET,
        index=False,
    )

    sample_df.to_csv(
        OUT_SAMPLE,
        index=False,
    )

    # ----------------------------------------------------------------------
    # 10. Features + modelos
    # ----------------------------------------------------------------------

    print(
        "\n[10/11] correlação feature × target..."
    )

    correlation_df = feature_target_correlation(
        gol_pool
    )

    correlation_df.to_csv(
        OUT_FEATURES,
        index=False,
    )

    print(
        md_table(correlation_df)
    )

    print(
        "\n[10b/11] teste controlado A_full × A_sem_DE..."
    )

    try:

        controlled_df, controlled_folds = (
            controlled_models(
                gol_pool
            )
        )

    except Exception as exc:

        print(
            f"      ERRO no CatBoost: "
            f"{type(exc).__name__}: {exc}"
        )

        controlled_df = pd.DataFrame()
        controlled_folds = pd.DataFrame()

    controlled_df.to_csv(
        OUT_CONTROLLED,
        index=False,
    )

    if not controlled_df.empty:

        print(
            md_table(controlled_df)
        )

    # ----------------------------------------------------------------------
    # 11. Diagnóstico
    # ----------------------------------------------------------------------

    print(
        "\n[11/11] construindo diagnóstico..."
    )

    diagnosis = build_diagnosis(
        raw_de_audit,
        first_summary,
        raw_vs_exp12,
        script01_comparison,
        controlled_df,
        correlation_df,
    )

    # ----------------------------------------------------------------------
    # Relatório
    # ----------------------------------------------------------------------

    sections = []

    # 1
    if official_candidates.empty:

        official_text = (
            "Nenhum candidato encontrado."
        )

    else:

        official_text = (
            "Os candidatos abaixo são apenas descoberta "
            "heurística. Nenhum foi declarado como pipeline "
            "oficial automaticamente.\n\n"
            + md_table(
                official_candidates.head(20)
            )
        )

    sections.append(
        (
            "1. Candidatos ao pipeline oficial",
            official_text,
        )
    )

    # 2
    sections.append(
        (
            "2. Target",
            md_table(target_df),
        )
    )

    # 3
    sections.append(
        (
            "3. Amostra e elegibilidade",
            md_table(sample_df),
        )
    )

    # 4
    sections.append(
        (
            "4. scout_DE RAW — natureza por temporada",
            md_table(raw_de_audit),
        )
    )

    # 5
    sections.append(
        (
            "5. Primeira aparição do scout_DE",
            md_table(first_summary)
            + "\n\n### Amostras grandes\n\n"
            + md_table(first_large),
        )
    )

    # 6
    sections.append(
        (
            "6. RAW → parquet EXP12",
            (
                md_table(
                    raw_vs_exp12[
                        [
                            "temporada",
                            "rodada",
                            "atleta_id",
                            "DE_raw",
                            "scout_DE",
                            "diff_raw_vs_exp12_scout_DE",
                            "abs_diff_raw_vs_exp12",
                        ]
                    ].sort_values(
                        "abs_diff_raw_vs_exp12",
                        ascending=False,
                    ).head(50)
                )
                if (
                    not raw_vs_exp12.empty
                    and "scout_DE" in raw_vs_exp12.columns
                )
                else "_Sem comparação disponível._"
            ),
        )
    )

    # 7
    sections.append(
        (
            "7. Script 01 real → gol_pool",
            (
                md_table(script01_comparison)
                + "\n\n### Status da execução\n\n"
                + f"`{script01_status}`"
            ),
        )
    )

    # 8
    sections.append(
        (
            "8. Trechos relevantes do Script 01",
            excerpt,
        )
    )

    # 9
    sections.append(
        (
            "9. Correlação feature × target",
            md_table(correlation_df),
        )
    )

    # 10
    sections.append(
        (
            "10. Teste controlado A_full × A_sem_DE",
            md_table(controlled_df),
        )
    )

    # 11
    sections.append(
        (
            "11. Diagnóstico",
            diagnosis,
        )
    )

    write_report(
        sections
    )

    # ----------------------------------------------------------------------
    # Manifest
    # ----------------------------------------------------------------------

    manifest = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),

        "paths": {
            "jog_exp12": str(JOG_EXP12),
            "jog_v2": str(JOG_V2),
            "gol_pool": str(GOL_POOL),
            "script01": str(SCRIPT01),
            "raw_root": str(RAW_ROOT),
        },

        "sha256": {
            "jog_exp12": sha256_file(
                JOG_EXP12
            ),
            "jog_v2": sha256_file(
                JOG_V2
            ),
            "gol_pool": sha256_file(
                GOL_POOL
            ),
            "script01": sha256_file(
                SCRIPT01
            ),
        },

        "script01_import": msg01,
        "script01_execution": script01_status,

        "shapes": {
            "jog_exp12": list(
                jog_exp12.shape
            ),
            "raw_gol": list(
                raw_gol.shape
            ),
            "gol_pool": list(
                gol_pool.shape
            ),
        },

        "official_candidates": (
            official_candidates.to_dict(
                orient="records"
            )
            if not official_candidates.empty
            else []
        ),

        "script01_comparison": (
            script01_summary
        ),

        "controlled_models": (
            controlled_df.to_dict(
                orient="records"
            )
            if not controlled_df.empty
            else []
        ),
    }

    OUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ----------------------------------------------------------------------
    # CSV adicional do RAW audit
    # ----------------------------------------------------------------------

    raw_de_audit.to_csv(
        OUT_DE_RAW,
        index=False,
    )

    first_summary.to_csv(
        OUT_DE_FIRST,
        index=False,
    )

    print("\n" + "=" * 80)
    print("AUDITORIA CONCLUÍDA")
    print("=" * 80)

    print(
        f"\nRelatório:"
        f"\n  {OUT_MD}"
    )

    print(
        "\nCSVs:"
        f"\n  {OUT_TARGET}"
        f"\n  {OUT_SAMPLE}"
        f"\n  {OUT_DE_RAW}"
        f"\n  {OUT_DE_FIRST}"
        f"\n  {OUT_DE_RAW_VS}"
        f"\n  {OUT_FEATURES}"
        f"\n  {OUT_CONTROLLED}"
        f"\n  {OUT_OFFICIAL}"
    )

    print(
        f"\nManifest:"
        f"\n  {OUT_MANIFEST}"
    )


if __name__ == "__main__":
    main()
