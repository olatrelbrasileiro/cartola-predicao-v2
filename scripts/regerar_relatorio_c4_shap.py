from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/workspaces/cartola-predicao-v2")
INPUT = ROOT / "reports/catboost_v2_shap_by_observation.parquet"

OUT_FI = ROOT / "reports/catboost_v2_shap_feature_importance.csv"
OUT_GI = ROOT / "reports/catboost_v2_shap_group_importance.csv"
OUT_CS = ROOT / "reports/catboost_v2_shap_com_vs_sem.csv"
OUT_MD = ROOT / "reports/catboost_v2_shap.md"

POS_NAMES = {
    1: "Goleiro",
    2: "Lateral",
    3: "Zagueiro",
    4: "Meia",
    5: "Atacante",
    6: "Tecnico",
}

FEATURES_COM = [
    "hist_mean",
    "hist_std",
    "hist_n",
    "hist_shrunk_k7",
    "rec_mean_5",
    "rec_mean_3",
    "rec_std_5",
    "momentum",
    "basic_mean",
    "basic_rec_5",
    "ga_per_game",
    "ga_rec_5",
    "ga_shrunk_k7",
    "entrou_rate",
    "is_cold_start",
    "posicao_id",
    "p_home",
    "p_draw",
    "p_away",
    "mando",
    "favoritismo_home_away",
]

FEATURES_SEM = [
    "hist_mean",
    "hist_std",
    "hist_n",
    "hist_shrunk_k7",
    "rec_mean_5",
    "rec_mean_3",
    "rec_std_5",
    "momentum",
    "basic_mean",
    "basic_rec_5",
    "ga_per_game",
    "ga_rec_5",
    "ga_shrunk_k7",
    "entrou_rate",
    "is_cold_start",
    "posicao_id",
]

GROUPS = {
    "historico": [
        "hist_mean",
        "hist_std",
        "hist_n",
        "hist_shrunk_k7",
    ],
    "recente": [
        "rec_mean_5",
        "rec_mean_3",
        "rec_std_5",
        "momentum",
    ],
    "basico_scouts": [
        "basic_mean",
        "basic_rec_5",
    ],
    "g_a": [
        "ga_per_game",
        "ga_rec_5",
        "ga_shrunk_k7",
    ],
    "oportunidade": [
        "entrou_rate",
        "is_cold_start",
    ],
    "posicao": [
        "posicao_id",
    ],
    "confronto": [
        "p_home",
        "p_draw",
        "p_away",
        "mando",
        "favoritismo_home_away",
    ],
}


def calc_stats(df, features, modelo, escopo, escopo_valor):
    rows = []

    for f in features:
        coluna_shap = "shap_posicao_id" if f == "posicao_id" else f
        s = df[coluna_shap].to_numpy(dtype=float)

        rows.append({
            "modelo": modelo,
            "escopo": escopo,
            "escopo_valor": escopo_valor,
            "feature": f,
            "n_obs": len(s),
            "mean_abs_shap": float(np.mean(np.abs(s))),
            "mean_shap": float(np.mean(s)),
            "std_shap": float(np.std(s)),
            "p25_shap": float(np.percentile(s, 25)),
            "p75_shap": float(np.percentile(s, 75)),
            "min_shap": float(np.min(s)),
            "max_shap": float(np.max(s)),
        })

    return rows


def gerar_estatisticas(df, features, modelo):
    rows = []

    # Global
    rows.extend(
        calc_stats(
            df,
            features,
            modelo,
            "global",
            "all",
        )
    )

    # Temporada
    for temporada, sub in df.groupby("temporada"):
        rows.extend(
            calc_stats(
                sub,
                features,
                modelo,
                "temporada",
                str(int(temporada)),
            )
        )

    # POSIÇÃO REAL
    for posicao_id, sub in df.groupby("posicao_id"):
        posicao_id = int(posicao_id)

        if posicao_id not in POS_NAMES:
            raise RuntimeError(
                f"posicao_id desconhecido no Parquet: {posicao_id}"
            )

        rows.extend(
            calc_stats(
                sub,
                features,
                modelo,
                "posicao",
                POS_NAMES[posicao_id],
            )
        )

    return rows


def agregar_grupos(df_fi):
    rows = []

    for (modelo, escopo, escopo_valor), sub in df_fi.groupby(
        ["modelo", "escopo", "escopo_valor"]
    ):
        total = float(sub["mean_abs_shap"].sum())

        for grupo, features in GROUPS.items():
            x = sub[sub["feature"].isin(features)]

            if len(x) == 0:
                continue

            soma = float(x["mean_abs_shap"].sum())

            rows.append({
                "modelo": modelo,
                "escopo": escopo,
                "escopo_valor": escopo_valor,
                "grupo": grupo,
                "n_features": int(x["n_obs"].sum()),
                "soma_mean_abs_shap": soma,
                "fracao_sobre_total": (
                    soma / total if total > 0 else np.nan
                ),
            })

    return pd.DataFrame(rows)


def gerar_com_vs_sem(df_fi):
    features_comuns = [
        f for f in FEATURES_COM
        if f in FEATURES_SEM
    ]

    rows = []

    for escopo in ["global", "temporada", "posicao"]:
        valores = df_fi.loc[
            df_fi["escopo"] == escopo,
            "escopo_valor"
        ].unique()

        for ev in valores:
            for feature in features_comuns:
                c = df_fi[
                    (df_fi["modelo"] == "COM")
                    & (df_fi["escopo"] == escopo)
                    & (df_fi["escopo_valor"] == ev)
                    & (df_fi["feature"] == feature)
                ]

                s = df_fi[
                    (df_fi["modelo"] == "SEM")
                    & (df_fi["escopo"] == escopo)
                    & (df_fi["escopo_valor"] == ev)
                    & (df_fi["feature"] == feature)
                ]

                if len(c) == 0 or len(s) == 0:
                    continue

                cval = float(c["mean_abs_shap"].iloc[0])
                sval = float(s["mean_abs_shap"].iloc[0])

                rows.append({
                    "escopo": escopo,
                    "escopo_valor": ev,
                    "feature": feature,
                    "com_mean_abs_shap": cval,
                    "sem_mean_abs_shap": sval,
                    "delta_abs": cval - sval,
                })

    return pd.DataFrame(rows)


def main():
    print("== REGERAÇÃO C4 — SHAP POR POSIÇÃO ==")

    df = pd.read_parquet(INPUT)

    print(f"Parquet: {INPUT}")
    print(f"shape: {df.shape}")

    # -----------------------------
    # Validação estrutural
    # -----------------------------
    obrigatorias = [
        "temporada",
        "rodada",
        "atleta_id",
        "posicao_id",
    ]

    for col in obrigatorias:
        if col not in df.columns:
            raise RuntimeError(f"Coluna ausente: {col}")

    posicoes = sorted(df["posicao_id"].unique())

    print("\n=== POSIÇÕES ===")
    print(
        df["posicao_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    if posicoes != [1, 2, 3, 4, 5, 6]:
        raise RuntimeError(
            f"Posições inválidas no Parquet: {posicoes}"
        )

    if df["posicao_id"].isna().any():
        raise RuntimeError("Há posicao_id nulo.")

    # -----------------------------
    # Montar DataFrames COM / SEM
    # -----------------------------
    com_cols = [
        "temporada",
        "rodada",
        "atleta_id",
        "posicao_id",
    ] + [
        f"com_{f}" for f in FEATURES_COM
    ]

    sem_cols = [
        "temporada",
        "rodada",
        "atleta_id",
        "posicao_id",
    ] + [
        f"sem_{f}" for f in FEATURES_SEM
    ]

    df_com = df[
        [
            "temporada",
            "rodada",
            "atleta_id",
            "posicao_id",
        ]
        + [
            f"com_{f}" for f in FEATURES_COM
        ]
    ].copy()

    df_sem = df[
        [
            "temporada",
            "rodada",
            "atleta_id",
            "posicao_id",
        ]
        + [
            f"sem_{f}" for f in FEATURES_SEM
        ]
    ].copy()

    # Remover prefixos COM/SEM.
    # IMPORTANTE:
    # - posicao_id = posição REAL da observação
    # - shap_posicao_id = contribuição SHAP da feature posicao_id
    # Não podemos ter duas colunas chamadas posicao_id.
    df_com.columns = (
        [
            "temporada",
            "rodada",
            "atleta_id",
            "posicao_id",
        ]
        + [
            "shap_posicao_id" if f == "posicao_id" else f
            for f in FEATURES_COM
        ]
    )

    df_sem.columns = (
        [
            "temporada",
            "rodada",
            "atleta_id",
            "posicao_id",
        ]
        + [
            "shap_posicao_id" if f == "posicao_id" else f
            for f in FEATURES_SEM
        ]
    )

    # A função de estatísticas espera que as features tenham seus
    # nomes canônicos. Para o cálculo do SHAP da feature posição,
    # usamos explicitamente shap_posicao_id sem alterar a posição real.

    # -----------------------------
    # Estatísticas
    # -----------------------------
    rows_fi = []

    rows_fi.extend(
        gerar_estatisticas(
            df_com,
            FEATURES_COM,
            "COM",
        )
    )

    rows_fi.extend(
        gerar_estatisticas(
            df_sem,
            FEATURES_SEM,
            "SEM",
        )
    )

    df_fi = pd.DataFrame(rows_fi)

    # -----------------------------
    # Grupos
    # -----------------------------
    df_gi = agregar_grupos(df_fi)

    # -----------------------------
    # COM vs SEM
    # -----------------------------
    df_cs = gerar_com_vs_sem(df_fi)

    # -----------------------------
    # Salvar
    # -----------------------------
    df_fi.to_csv(OUT_FI, index=False)
    df_gi.to_csv(OUT_GI, index=False)
    df_cs.to_csv(OUT_CS, index=False)

    # -----------------------------
    # Validação final
    # -----------------------------
    pos_fi = sorted(
        df_fi.loc[
            df_fi["escopo"] == "posicao",
            "escopo_valor"
        ].unique()
    )

    esperadas = [
        "Goleiro",
        "Lateral",
        "Zagueiro",
        "Meia",
        "Atacante",
        "Tecnico",
    ]

    posicao_ok = set(pos_fi) == set(esperadas)

    if not posicao_ok:
        raise RuntimeError(
            f"Posições no relatório incorretas: {pos_fi}"
        )

    # Cada modelo deve ter 6 posições × n_features
    n_com = len(df_fi[
        (df_fi["modelo"] == "COM")
        & (df_fi["escopo"] == "posicao")
    ])

    n_sem = len(df_fi[
        (df_fi["modelo"] == "SEM")
        & (df_fi["escopo"] == "posicao")
    ])

    if n_com != 6 * len(FEATURES_COM):
        raise RuntimeError(
            f"Quantidade inesperada COM por posição: {n_com}"
        )

    if n_sem != 6 * len(FEATURES_SEM):
        raise RuntimeError(
            f"Quantidade inesperada SEM por posição: {n_sem}"
        )

    # -----------------------------
    # Markdown resumido
    # -----------------------------
    linhas = []

    linhas.append("# C4 — SHAP CatBoost v2\n\n")
    linhas.append(
        "Relatório regenerado a partir do Parquet SHAP persistido. "
        "Nenhum modelo foi treinado novamente.\n\n"
    )

    linhas.append("## Integridade\n\n")
    linhas.append(f"- Observações: {len(df)}\n")
    linhas.append(f"- Shape: {df.shape}\n")
    linhas.append(f"- Posições válidas: {posicoes}\n")
    linhas.append(
        f"- Relatório por posição: "
        f"{'PASS' if posicao_ok else 'FAIL'}\n"
    )
    linhas.append(
        f"- COM por posição: "
        f"{n_com} linhas\n"
    )
    linhas.append(
        f"- SEM por posição: "
        f"{n_sem} linhas\n\n"
    )

    linhas.append("## Importância global por grupo\n\n")

    global_gi = df_gi[
        df_gi["escopo"] == "global"
    ].copy()

    for modelo in ["COM", "SEM"]:
        linhas.append(f"### {modelo}\n\n")
        sub = global_gi[
            global_gi["modelo"] == modelo
        ].sort_values(
            "fracao_sobre_total",
            ascending=False,
        )

        for _, r in sub.iterrows():
            linhas.append(
                f"- {r['grupo']}: "
                f"{r['fracao_sobre_total']:.4%}\n"
            )

        linhas.append("\n")

    linhas.append("## Grupos por posição — COM\n\n")

    pos_gi = df_gi[
        (df_gi["modelo"] == "COM")
        & (df_gi["escopo"] == "posicao")
    ].copy()

    for posicao in esperadas:
        linhas.append(f"### {posicao}\n\n")

        sub = pos_gi[
            pos_gi["escopo_valor"] == posicao
        ].sort_values(
            "fracao_sobre_total",
            ascending=False,
        )

        for _, r in sub.iterrows():
            linhas.append(
                f"- {r['grupo']}: "
                f"{r['fracao_sobre_total']:.4%}\n"
            )

        linhas.append("\n")

    OUT_MD.write_text(
        "".join(linhas),
        encoding="utf-8",
    )

    print("\n=== RESULTADO ===")
    print("PASS")
    print(f"FI: {OUT_FI}")
    print(f"GI: {OUT_GI}")
    print(f"CS: {OUT_CS}")
    print(f"MD: {OUT_MD}")


if __name__ == "__main__":
    main()
