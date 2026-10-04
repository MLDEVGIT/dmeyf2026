#!/usr/bin/env python3
"""
z517_competencia_01_optuna.py

Optimización offline de hiperparámetros LightGBM para Competencia 01.

Diseño principal:
    TRAIN: 202103, 202104, 202105
    TEST : 202106

Target:
    BAJA+2 = 1
    BAJA+1 / CONTINUA = 0

Objetivo Optuna:
    maximizar ganancia económica en TEST 202106.

La búsqueda replica los cinco hiperparámetros trabajados en z494:
    - num_iterations
    - learning_rate
    - feature_fraction
    - num_leaves
    - min_data_in_leaf

IMPORTANTE:
    - No utiliza 202107.
    - No utiliza 202108.
    - No genera submits.
    - No modifica z515/z516.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


# ============================================================
# Configuración
# ============================================================

DATASET_DEFAULT = Path("/data/dmeyf/datasets/competencia_01.csv")

OUTPUT_DEFAULT = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/competencia_01/optuna_z517"
)

TRAIN_MONTHS = [202103, 202104, 202105]
TEST_MONTH = 202106

TARGET = "clase_ternaria"
ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"

GANANCIA_BAJA2 = 1_072_500
COSTO_ESTIMULO = -27_500

SEMILLA = 290497

# Rango amplio para estudiar la curva de ganancia.
# Incluye toda la zona relevante que ya observamos con z515.
CORTES = np.arange(4000, 19001, 500, dtype=int)

# Baseline z515
BASELINE = {
    "num_iterations": 1200,
    "learning_rate": 0.02,
    "feature_fraction": 0.5,
    "num_leaves": 750,
    "min_data_in_leaf": 5000,
}

BASELINE_GAIN_202106 = 398_200_000


# ============================================================
# Argumentos
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna + LightGBM para Competencia 01"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DEFAULT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DEFAULT,
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Cantidad de trials de Optuna.",
    )

    return parser.parse_args()


# ============================================================
# Utilidades
# ============================================================

def fmt_millones(x: float) -> str:
    return f"{x / 1_000_000:.3f} M"


def calcular_ganancias(
    y_true: np.ndarray,
    prob: np.ndarray,
    cortes: np.ndarray,
) -> pd.DataFrame:
    """
    Ordena por probabilidad descendente y calcula ganancia
    acumulada para cada corte.
    """

    orden = np.argsort(-prob)
    y_ord = y_true[orden]

    ganancias_individuales = np.where(
        y_ord == 1,
        GANANCIA_BAJA2,
        COSTO_ESTIMULO,
    )

    acumulada = np.cumsum(ganancias_individuales)

    filas = []

    for corte in cortes:
        if corte > len(y_ord):
            continue

        y_top = y_ord[:corte]

        positivos = int(y_top.sum())
        precision = positivos / corte
        total_positivos = int(y_true.sum())

        recall = (
            positivos / total_positivos
            if total_positivos > 0
            else np.nan
        )

        ganancia = int(acumulada[corte - 1])

        filas.append(
            {
                "corte": int(corte),
                "positivos": positivos,
                "precision": precision,
                "recall": recall,
                "ganancia": ganancia,
            }
        )

    return pd.DataFrame(filas)


def mejor_corte(
    y_true: np.ndarray,
    prob: np.ndarray,
) -> tuple[pd.Series, pd.DataFrame]:

    tabla = calcular_ganancias(
        y_true=y_true,
        prob=prob,
        cortes=CORTES,
    )

    idx = tabla["ganancia"].idxmax()
    mejor = tabla.loc[idx]

    return mejor, tabla


# ============================================================
# Lectura
# ============================================================

def cargar_datos(dataset_path: Path):
    """
    Usa DuckDB para evitar los problemas que observamos
    previamente con pandas.read_csv(usecols=...).
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No existe el dataset: {dataset_path}"
        )

    meses_sql = ", ".join(
        str(x) for x in TRAIN_MONTHS + [TEST_MONTH]
    )

    query = f"""
        SELECT *
        FROM read_csv_auto(
            '{dataset_path}',
            header = true,
            sample_size = -1
        )
        WHERE foto_mes IN ({meses_sql})
    """

    print("\nCargando datos con DuckDB...")
    inicio = time.time()

    con = duckdb.connect()
    try:
        df = con.execute(query).fetchdf()
    finally:
        con.close()

    print(
        f"Datos cargados: {len(df):,} filas "
        f"en {time.time() - inicio:.1f} s"
    )

    print("\nFilas por mes:")
    print(
        df.groupby(MONTH_COL)
        .size()
        .to_string()
    )

    return df


# ============================================================
# Preparación
# ============================================================

def preparar_datos(df: pd.DataFrame):
    train = df[df[MONTH_COL].isin(TRAIN_MONTHS)].copy()
    test = df[df[MONTH_COL] == TEST_MONTH].copy()

    if train[TARGET].isna().any():
        raise ValueError(
            "TRAIN contiene clase_ternaria faltante."
        )

    if test[TARGET].isna().any():
        raise ValueError(
            "TEST contiene clase_ternaria faltante."
        )

    feature_cols = [
        c
        for c in df.columns
        if c not in {ID_COL, MONTH_COL, TARGET}
    ]

    print(f"\nTRAIN meses : {TRAIN_MONTHS}")
    print(f"TEST mes    : {TEST_MONTH}")

    print(f"\nTRAIN filas : {len(train):,}")
    print(f"TEST filas  : {len(test):,}")
    print(f"Variables   : {len(feature_cols)}")

    y_train = (
        train[TARGET]
        .eq("BAJA+2")
        .astype(np.int8)
        .to_numpy()
    )

    y_test = (
        test[TARGET]
        .eq("BAJA+2")
        .astype(np.int8)
        .to_numpy()
    )

    print(
        f"\nTRAIN positivos BAJA+2: "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"TEST positivos BAJA+2 : "
        f"{int(y_test.sum()):,}"
    )

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    return X_train, y_train, X_test, y_test, feature_cols


# ============================================================
# Baseline
# ============================================================

def evaluar_baseline(
    X_train,
    y_train,
    X_test,
    y_test,
):
    print("\n" + "=" * 70)
    print("BASELINE z515")
    print("=" * 70)

    modelo = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=BASELINE["num_iterations"],
        learning_rate=BASELINE["learning_rate"],
        num_leaves=BASELINE["num_leaves"],
        max_depth=-1,
        min_child_samples=BASELINE["min_data_in_leaf"],
        max_bin=31,
        colsample_bytree=BASELINE["feature_fraction"],
        subsample=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=SEMILLA,
        n_jobs=-1,
        verbosity=-1,
    )

    inicio = time.time()

    modelo.fit(X_train, y_train)

    prob = modelo.predict_proba(X_test)[:, 1]

    elapsed = time.time() - inicio

    auc = roc_auc_score(y_test, prob)
    ap = average_precision_score(y_test, prob)
    ll = log_loss(y_test, prob)

    mejor, tabla = mejor_corte(y_test, prob)

    print(f"AUC       : {auc:.6f}")
    print(f"AP        : {ap:.6f}")
    print(f"LogLoss   : {ll:.6f}")
    print(f"Mejor N   : {int(mejor['corte'])}")
    print(
        f"Ganancia  : "
        f"{fmt_millones(mejor['ganancia'])}"
    )
    print(f"Tiempo    : {elapsed:.1f} s")

    diferencia = (
        mejor["ganancia"] - BASELINE_GAIN_202106
    )

    print(
        "Diferencia vs referencia z515: "
        f"{fmt_millones(diferencia)}"
    )

    return {
        "auc": auc,
        "ap": ap,
        "logloss": ll,
        "corte": int(mejor["corte"]),
        "ganancia": int(mejor["ganancia"]),
        "tabla": tabla,
    }


# ============================================================
# Optuna
# ============================================================

def crear_objective(
    X_train,
    y_train,
    X_test,
    y_test,
    resultados: list[dict],
):

    def objective(trial: optuna.Trial):

        params = {
            "num_iterations": trial.suggest_int(
                "num_iterations",
                8,
                2048,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.30,
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction",
                0.10,
                1.00,
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves",
                8,
                2048,
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf",
                1,
                8000,
            ),
        }

        modelo = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=params["num_iterations"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=-1,
            min_child_samples=params["min_data_in_leaf"],
            max_bin=31,
            colsample_bytree=params["feature_fraction"],
            subsample=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=SEMILLA,
            n_jobs=-1,
            verbosity=-1,
        )

        inicio = time.time()

        modelo.fit(X_train, y_train)

        prob = modelo.predict_proba(X_test)[:, 1]

        elapsed = time.time() - inicio

        auc = roc_auc_score(y_test, prob)
        ap = average_precision_score(y_test, prob)
        ll = log_loss(y_test, prob)

        mejor, _ = mejor_corte(
            y_true=y_test,
            prob=prob,
        )

        ganancia = int(mejor["ganancia"])
        corte = int(mejor["corte"])

        trial.set_user_attr("auc", float(auc))
        trial.set_user_attr("ap", float(ap))
        trial.set_user_attr("logloss", float(ll))
        trial.set_user_attr("corte", corte)
        trial.set_user_attr(
            "tiempo_segundos",
            float(elapsed),
        )

        resultados.append(
            {
                "trial": trial.number,
                **params,
                "auc": auc,
                "ap": ap,
                "logloss": ll,
                "corte": corte,
                "ganancia": ganancia,
                "tiempo_segundos": elapsed,
            }
        )

        print(
            f"\nTrial {trial.number:03d} | "
            f"gain={fmt_millones(ganancia)} | "
            f"N={corte:5d} | "
            f"AUC={auc:.6f} | "
            f"AP={ap:.6f} | "
            f"{elapsed:.1f}s"
        )

        return ganancia

    return objective


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("z517 - OPTIMIZACION LIGHTGBM / COMPETENCIA 01")
    print("=" * 70)

    print(f"\nDataset : {args.dataset}")
    print(f"Output  : {args.output_dir}")
    print(f"Trials  : {args.trials}")
    print(f"Semilla : {SEMILLA}")

    df = cargar_datos(args.dataset)

    (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_cols,
    ) = preparar_datos(df)

    # --------------------------------------------------------
    # Baseline reproducido dentro de z517
    # --------------------------------------------------------

    baseline = evaluar_baseline(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    baseline["tabla"].to_csv(
        args.output_dir / "baseline_curva_ganancia.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Optuna
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INICIANDO OPTUNA")
    print("=" * 70)

    sampler = optuna.samplers.TPESampler(
        seed=SEMILLA,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="competencia_01_z517",
    )

    resultados: list[dict] = []

    objective = crear_objective(
        X_train,
        y_train,
        X_test,
        y_test,
        resultados,
    )

    inicio_optuna = time.time()

    study.optimize(
        objective,
        n_trials=args.trials,
        gc_after_trial=True,
    )

    tiempo_total = time.time() - inicio_optuna

    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    resultados_df = pd.DataFrame(resultados)

    resultados_df = resultados_df.sort_values(
        ["ganancia", "auc"],
        ascending=[False, False],
    )

    resultados_path = (
        args.output_dir / "trials_z517.csv"
    )

    resultados_df.to_csv(
        resultados_path,
        index=False,
    )

    best = study.best_trial

    resumen = {
        "train_months": TRAIN_MONTHS,
        "test_month": TEST_MONTH,
        "n_features": len(feature_cols),
        "n_trials": args.trials,
        "seed": SEMILLA,
        "baseline": {
            "params": BASELINE,
            "auc": baseline["auc"],
            "ap": baseline["ap"],
            "logloss": baseline["logloss"],
            "corte": baseline["corte"],
            "ganancia": baseline["ganancia"],
        },
        "best_trial": {
            "number": best.number,
            "value": int(best.value),
            "params": best.params,
            "user_attrs": best.user_attrs,
        },
        "tiempo_total_segundos": tiempo_total,
    }

    with open(
        args.output_dir / "resumen_z517.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            resumen,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("RESULTADO FINAL z517")
    print("=" * 70)

    print(
        f"\nBaseline : "
        f"{fmt_millones(baseline['ganancia'])}"
    )

    print(
        f"Mejor BO : "
        f"{fmt_millones(best.value)}"
    )

    mejora = best.value - baseline["ganancia"]

    print(
        f"Mejora   : "
        f"{fmt_millones(mejora)}"
    )

    print(
        f"\nBest trial: {best.number}"
    )

    print("\nHiperparámetros:")

    for k, v in best.params.items():
        print(f"  {k}: {v}")

    print("\nMétricas:")
    print(
        f"  AUC     : "
        f"{best.user_attrs['auc']:.6f}"
    )
    print(
        f"  AP      : "
        f"{best.user_attrs['ap']:.6f}"
    )
    print(
        f"  LogLoss : "
        f"{best.user_attrs['logloss']:.6f}"
    )
    print(
        f"  corte   : "
        f"{best.user_attrs['corte']}"
    )

    print(
        f"\nTiempo Optuna: "
        f"{tiempo_total / 60:.1f} minutos"
    )

    print(
        f"\nResultados completos:\n"
        f"  {resultados_path}"
    )

    print(
        f"  {args.output_dir / 'resumen_z517.json'}"
    )

    print("\nTOP 10:")
    columnas_top = [
        "trial",
        "ganancia",
        "corte",
        "auc",
        "ap",
        "num_iterations",
        "learning_rate",
        "feature_fraction",
        "num_leaves",
        "min_data_in_leaf",
    ]

    print(
        resultados_df[columnas_top]
        .head(10)
        .to_string(index=False)
    )

    print("\nIMPORTANTE:")
    print(
        "El ganador de 202106 todavía NO debe considerarse "
        "modelo final."
    )
    print(
        "Debe validarse posteriormente sobre 202105 antes "
        "de utilizar 202108."
    )


if __name__ == "__main__":
    main()