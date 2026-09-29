#!/usr/bin/env python3
"""
Auditoria estrutural (read-only) — confirma que o experimental pode
importar catboost_v2 SEM duplicar constantes e SEM executar main().
Nao altera nada.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def detect_root():
    for c in [Path.home() / "cartola-predicao-v2",
              Path("/workspaces/cartola-predicao-v2"),
              Path(__file__).resolve().parent.parent]:
        if (c / "reports" / "harness_folds_v2.csv").exists():
            return c
    raise RuntimeError("Raiz nao encontrada")


ROOT = detect_root()
CB_V2_PATH = ROOT / "scripts" / "catboost_v2.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def h(obj):
    """Hash canonico de um objeto Python serializavel."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def main():
    print("== Auditoria estrutural — design de ablacao ==")
    print(f"catboost_v2: {CB_V2_PATH}")

    cbv2 = load_module("cbv2_audit", CB_V2_PATH)

    # ---- 1. constantes publicas que o experimental vai importar
    print("\n[1] Constantes publicas do catboost_v2")
    checks = {
        "CATBOOST_PARAMS": h(cbv2.CATBOOST_PARAMS),
        "FEATURES_COM":    h(cbv2.FEATURES_COM),
        "FEATURES_SEM":    h(cbv2.FEATURES_SEM),
        "CAT_FEATURES":    h(cbv2.CAT_FEATURES),
        "TARGET":          cbv2.TARGET,
        "HASH_CANON":      cbv2.HASH_CANON[:16] + "...",
        "HASH_HARNESS":    cbv2.HASH_HARNESS[:16] + "...",
        "HASH_P0":         cbv2.HASH_P0[:16] + "...",
    }
    for k, v in checks.items():
        print(f"  {k:16s} {v}")

    # ---- 2. features nao contem status_id nem elegivel
    print("\n[2] Sanidade das listas de features")
    for name in ["FEATURES_COM", "FEATURES_SEM"]:
        lst = getattr(cbv2, name)
        proibidas = [c for c in lst if c in ("status_id", "elegivel")]
        print(f"  {name:14s} len={len(lst):2d}  proibidas_presentes={proibidas}")
        if proibidas:
            print(f"  *** ERRO: {name} contem {proibidas}. Abortando.", file=sys.stderr)
            sys.exit(2)

    # ---- 3. funcoes publicas necessarias
    print("\n[3] Funcoes publicas necessarias")
    fns = ["fit_predict_catboost", "mae", "rmse", "spearman",
           "sha256", "load_yaml", "_load_ridge_v2"]
    for f in fns:
        ok = hasattr(cbv2, f) and callable(getattr(cbv2, f))
        print(f"  {f:24s} {'OK' if ok else '*** AUSENTE ***'}")
        if not ok:
            sys.exit(2)

    # ---- 4. rv2 ja carregado no import (efeito colateral esperado)
    print("\n[4] rv2 carregado como efeito colateral do import")
    print(f"  rv2: {type(cbv2.rv2).__name__}")
    for f in ["compute_priors", "add_shrunk_features", "apply_cumulative_diff",
              "compute_basic_score", "compute_participation_features"]:
        ok = hasattr(cbv2.rv2, f) and callable(getattr(cbv2.rv2, f))
        print(f"  rv2.{f:32s} {'OK' if ok else '*** AUSENTE ***'}")
        if not ok:
            sys.exit(2)

    # ---- 5. main() NAO foi executado no import
    print("\n[5] main() NAO executado no import")
    n_fits = cbv2.FIT_COUNTER.get("n_catboost_fits", -1)
    print(f"  FIT_COUNTER['n_catboost_fits'] = {n_fits}")
    if n_fits != 0:
        print("  *** ERRO: import disparou fits. Revisar module-level code.",
              file=sys.stderr)
        sys.exit(2)

    # ---- 6. assinaturas (leitura pura via inspect)
    import inspect
    print("\n[6] Assinaturas (para conferir chamadas do experimental)")
    for f in ["fit_predict_catboost", "mae", "rmse", "spearman"]:
        try:
            sig = inspect.signature(getattr(cbv2, f))
            print(f"  cbv2.{f}{sig}")
        except (TypeError, ValueError):
            print(f"  cbv2.{f} <sem assinatura>")
    for f in ["compute_priors", "add_shrunk_features",
              "apply_cumulative_diff", "compute_basic_score",
              "compute_participation_features"]:
        try:
            sig = inspect.signature(getattr(cbv2.rv2, f))
            print(f"  rv2.{f}{sig}")
        except (TypeError, ValueError):
            print(f"  rv2.{f} <sem assinatura>")

    # ---- 7. anexar hash dos objetos auditados a um JSON
    out = ROOT / "reports" / "audit_ablation_design.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump({
            "catboost_v2_path": str(CB_V2_PATH),
            "hashes": checks,
            "features_com_len": len(cbv2.FEATURES_COM),
            "features_sem_len": len(cbv2.FEATURES_SEM),
            "cat_features": cbv2.CAT_FEATURES,
            "target": cbv2.TARGET,
            "main_nao_executado": n_fits == 0,
        }, fp, indent=2, ensure_ascii=False)
    print(f"\nartefato: {out}")
    print("OK — design valido para ablacao por import.")


if __name__ == "__main__":
    main()
