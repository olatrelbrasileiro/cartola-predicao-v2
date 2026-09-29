from pathlib import Path
import json
import pandas as pd

ROOT = Path(".")
API_DIR = ROOT / "experiments/data/cartola_api_2026"
CSV_DIR = ROOT / "data/raw/caRtola/2026"
OUT_DIR = ROOT / "reports/exp12/api_vs_csv_2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COMPARADOR API × CSV — CARTOLA 2026")
print("=" * 80)
print()

SCOUTS = [
    "A", "CA", "CV", "DE", "DS", "FC", "FD", "FF",
    "FS", "FT", "G", "GC", "GS", "I", "PC", "PS", "SG", "V"
]

divergencias = []
resumos = []
only_api_rows = []
only_csv_rows = []

def same_num(a, b):
    # NaN/None equivalentes
    if pd.isna(a) and pd.isna(b):
        return True

    try:
        return abs(float(a) - float(b)) <= 1e-9
    except Exception:
        return str(a) == str(b)

for rodada in range(1, 29):

    api_path = API_DIR / f"r{rodada}.json"
    csv_path = CSV_DIR / f"rodada-{rodada}.csv"

    print(f"R{rodada:02d}: ", end="", flush=True)

    with open(api_path, encoding="utf-8") as f:
        api = json.load(f)

    csv = pd.read_csv(csv_path)

    # API /atletas/pontuados retorna dict: atleta_id -> dados
    atletas_api = api["atletas"]

    api_by_id = {}
    for key, atleta in atletas_api.items():
        aid = int(key)
        api_by_id[aid] = atleta

    csv_by_id = {}

    for _, row in csv.iterrows():
        try:
            aid = int(row["atletas.atleta_id"])
            csv_by_id[aid] = row
        except Exception:
            pass

    api_ids = set(api_by_id)
    csv_ids = set(csv_by_id)

    only_api = api_ids - csv_ids
    only_csv = csv_ids - api_ids

    for aid in only_api:
        only_api_rows.append({
            "rodada": rodada,
            "atleta_id": aid
        })

    for aid in only_csv:
        only_csv_rows.append({
            "rodada": rodada,
            "atleta_id": aid
        })

    # Comparar apenas interseção
    comum = api_ids & csv_ids

    n_part = 0
    n_club = 0
    n_pos = 0
    n_pts = 0
    n_scout = 0
    n_any = 0

    for aid in comum:

        a = api_by_id[aid]
        r = csv_by_id[aid]

        api_part = bool(a.get("entrou_em_campo", False))
        csv_part = bool(r["atletas.entrou_em_campo"])

        api_club = a.get("clube_id")
        csv_club = r["atletas.clube_id"]

        api_pos = a.get("posicao_id")
        csv_pos = r["atletas.posicao_id"]

        api_pts = a.get("pontuacao")
        csv_pts = r["atletas.pontos_num"]

        part_diff = api_part != csv_part
        club_diff = not same_num(api_club, csv_club)
        pos_diff = not same_num(api_pos, csv_pos)
        pts_diff = not same_num(api_pts, csv_pts)

        scout_diffs = {}

        # API scout é dict; CSV usa uma coluna por scout.
        api_scout = a.get("scout") or {}

        for scout in SCOUTS:
            api_value = api_scout.get(scout, 0)

            csv_value = r.get(scout)

            # NaN no CSV significa ausência daquele scout.
            if pd.isna(csv_value):
                csv_value = 0

            try:
                api_value = float(api_value)
                csv_value = float(csv_value)
            except Exception:
                pass

            if not same_num(api_value, csv_value):
                scout_diffs[scout] = {
                    "api": api_value,
                    "csv": csv_value
                }

        scout_diff = len(scout_diffs) > 0

        if part_diff:
            n_part += 1

        if club_diff:
            n_club += 1

        if pos_diff:
            n_pos += 1

        if pts_diff:
            n_pts += 1

        if scout_diff:
            n_scout += 1

        if part_diff or club_diff or pos_diff or pts_diff or scout_diff:
            n_any += 1

            divergencias.append({
                "rodada": rodada,
                "atleta_id": aid,
                "participacao_diff": part_diff,
                "clube_diff": club_diff,
                "posicao_diff": pos_diff,
                "pontos_diff": pts_diff,
                "scout_diff": scout_diff,
                "api_entrou_em_campo": api_part,
                "csv_entrou_em_campo": csv_part,
                "api_clube_id": api_club,
                "csv_clube_id": csv_club,
                "api_posicao_id": api_pos,
                "csv_posicao_id": csv_pos,
                "api_pontuacao": api_pts,
                "csv_pontuacao": csv_pts,
                "scout_diferencas": json.dumps(
                    scout_diffs,
                    ensure_ascii=False,
                    sort_keys=True
                )
            })

    resumos.append({
        "rodada": rodada,
        "api_atletas": len(api_ids),
        "csv_atletas": len(csv_ids),
        "intersecao": len(comum),
        "only_api": len(only_api),
        "only_csv": len(only_csv),
        "divergencias": n_any,
        "participacao_diff": n_part,
        "clube_diff": n_club,
        "posicao_diff": n_pos,
        "pontos_diff": n_pts,
        "scout_diff": n_scout,
    })

    print(
        f"API={len(api_ids)} | "
        f"CSV={len(csv_ids)} | "
        f"comum={len(comum)} | "
        f"API-only={len(only_api)} | "
        f"CSV-only={len(only_csv)} | "
        f"div={n_any}"
    )

pd.DataFrame(resumos).to_csv(
    OUT_DIR / "por_rodada.csv",
    index=False
)

pd.DataFrame(divergencias).to_csv(
    OUT_DIR / "divergencias_detalhadas.csv",
    index=False
)

pd.DataFrame(only_api_rows).to_csv(
    OUT_DIR / "only_api.csv",
    index=False
)

pd.DataFrame(only_csv_rows).to_csv(
    OUT_DIR / "only_csv.csv",
    index=False
)

print()
print("=" * 80)
print("COMPARAÇÃO CONCLUÍDA")
print("=" * 80)
print(f"Rodadas: 1-28")
print(f"Divergências detalhadas: {len(divergencias)}")
print(f"Saída: {OUT_DIR}")
print()
print("Arquivos:")
print("  por_rodada.csv")
print("  divergencias_detalhadas.csv")
print("  only_api.csv")
print("  only_csv.csv")
