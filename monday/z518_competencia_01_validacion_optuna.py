#!/usr/bin/env python3
"""
z518_competencia_01_validacion_optuna.py

Validación temporal independiente de los hiperparámetros obtenidos
por Optuna en z517.

Diseño:
    TRAIN: 202103, 202104
    TEST : 202105

Compara:
    1. Baseline z515
    2. Trial 46 ganador de z517, SIN volver a optimizar

Target:
    BAJA+2 = 1
    BAJA+1 / CONTINUA = 0

IMPORTANTE:
    - No utiliza 202106 para seleccionar parámetros.
    - No utiliza 202107.
    - No utiliza 202108.
    - No ejecuta Optuna.
    - No genera submits.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


# ============================================================
# Configuración
# ============================================================

DATASET = Path("/data/dmeyf/datasets/competencia_01.csv")

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/validacion_z518"
)

TRAIN_MONTHS = [202103, 202104]
TEST_MONTH = 202105

TARGET = "clase_ternaria"
ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"

GANANCIA_BAJA2 = 1_072_500
COSTO_ESTIMULO = -27_500

SEMILLA = 290497

CORTES = np.arange(4000, 19001, 500, dtype=int)


# ============================================================
# Parámetros
# ============================================================

BASELINE_PARAMS = {
    "num_iterations": 1200,
    "learning_rate": 0.02,
    "feature_fraction": 0.5,
    "num_leaves": 750,
    "min_data_in_leaf": 5000,
}

# Ganador de z517.
# Estos parámetros quedan CONGELADOS.
TRIAL46_PARAMS = {
    "num_iterations": 1828,
    "learning_rate": 0.011573579444249976,
    "feature_fraction": 0.9178523224363775,
    "num_leaves": 665,
    "min_data_in_leaf": 1261,
}

# Resultado histórico de z515 para esta ventana temporal.
BASELINE_REFERENCIA_GAIN = 248_050_000
BASELINE_REFERENCIA_AUC = 0.882142
BASELINE_REFERENCIA_AP = 0.050176
BASELINE_REFERENCIA_CORTE = 8500


# ============================================================
# Utilidades
# ============================================================

def fmt_millones(x: float) -> str:
    return f"{x / 1_000_000:.3f} M"


def calcular_ganancias(
    y_true: np.ndarray,
    prob: np.ndarray,
) -> pd.DataFrame:

    orden = np.argsort(-prob)
    y_ord = y_true[orden]

    ganancias_individuales = np.where(
        y_ord == 1,
        GANANCIA_BAJA2,
        COSTO_ESTIMULO,
    )

    acumulada = np.cumsum(ganancias_individuales)

    filas = []

    for corte in CORTES:
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


def evaluar_modelo(
    nombre: str,
    params: dict,
    X_train,
    y_train,
    X_test,
    y_test,
):
    print("\n" + "=" * 70)
    print(nombre)
    print("=" * 70)

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

    tabla = calcular_ganancias(
        y_true=y_test,
        prob=prob,
    )

    idx = tabla["ganancia"].idxmax()
    mejor = tabla.loc[idx]

    print(f"AUC       : {auc:.6f}")
    print(f"AP        : {ap:.6f}")
    print(f"LogLoss   : {ll:.6f}")
    print(f"Mejor N   : {int(mejor['corte'])}")
    print(
        f"Positivos : {int(mejor['positivos'])}"
    )
    print(
        f"Precision : {mejor['precision']:.4f}"
    )
    print(
        f"Recall    : {mejor['recall']:.4f}"
    )
    print(
        f"Ganancia  : {fmt_millones(mejor['ganancia'])}"
    )
    print(f"Tiempo    : {elapsed:.1f} s")

    return {
        "nombre": nombre,
        "auc": float(auc),
        "ap": float(ap),
        "logloss": float(ll),
        "corte": int(mejor["corte"]),
        "positivos": int(mejor["positivos"]),
        "precision": float(mejor["precision"]),
        "recall": float(mejor["recall"]),
        "ganancia": int(mejor["ganancia"]),
        "tiempo_segundos": float(elapsed),
        "tabla": tabla,
    }


# ============================================================
# Lectura
# ============================================================

def cargar_datos():
    meses = TRAIN_MONTHS + [TEST_MONTH]

    meses_sql = ", ".join(str(x) for x in meses)

    query = f"""
        SELECT *
        FROM read_csv_auto(
            '{DATASET}',
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
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("z518 - VALIDACION TEMPORAL DE HIPERPARAMETROS")
    print("=" * 70)

    print(f"\nDataset : {DATASET}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"TRAIN   : {TRAIN_MONTHS}")
    print(f"TEST    : {TEST_MONTH}")
    print(f"Semilla : {SEMILLA}")

    df = cargar_datos()

    train = df[
        df[MONTH_COL].isin(TRAIN_MONTHS)
    ].copy()

    test = df[
        df[MONTH_COL] == TEST_MONTH
    ].copy()

    if train[TARGET].isna().any():
        raise ValueError(
            "TRAIN contiene target faltante."
        )

    if test[TARGET].isna().any():
        raise ValueError(
            "TEST contiene target faltante."
        )

    feature_cols = [
        c
        for c in df.columns
        if c not in {ID_COL, MONTH_COL, TARGET}
    ]

    X_train = train[feature_cols]
    X_test = test[feature_cols]

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

    print(f"\nTRAIN filas : {len(train):,}")
    print(f"TEST filas  : {len(test):,}")
    print(f"Variables   : {len(feature_cols)}")

    print(
        f"TRAIN BAJA+2: {int(y_train.sum()):,}"
    )

    print(
        f"TEST BAJA+2 : {int(y_test.sum()):,}"
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = evaluar_modelo(
        nombre="BASELINE z515",
        params=BASELINE_PARAMS,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    baseline["tabla"].to_csv(
        OUTPUT_DIR / "curva_baseline.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Trial 46
    # --------------------------------------------------------

    trial46 = evaluar_modelo(
        nombre="TRIAL 46 z517 - PARAMETROS CONGELADOS",
        params=TRIAL46_PARAMS,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    trial46["tabla"].to_csv(
        OUTPUT_DIR / "curva_trial46.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Comparación
    # --------------------------------------------------------

    mejora_gain = (
        trial46["ganancia"]
        - baseline["ganancia"]
    )

    mejora_pct = (
        100.0
        * mejora_gain
        / baseline["ganancia"]
    )

    delta_auc = (
        trial46["auc"]
        - baseline["auc"]
    )

    delta_ap = (
        trial46["ap"]
        - baseline["ap"]
    )

    print("\n" + "=" * 70)
    print("COMPARACION OOT")
    print("=" * 70)

    print(
        f"\n{'Modelo':<20}"
        f"{'Ganancia':>15}"
        f"{'N':>8}"
        f"{'AUC':>12}"
        f"{'AP':>12}"
    )

    print("-" * 67)

    print(
        f"{'Baseline':<20}"
        f"{fmt_millones(baseline['ganancia']):>15}"
        f"{baseline['corte']:>8}"
        f"{baseline['auc']:>12.6f}"
        f"{baseline['ap']:>12.6f}"
    )

    print(
        f"{'Trial 46':<20}"
        f"{fmt_millones(trial46['ganancia']):>15}"
        f"{trial46['corte']:>8}"
        f"{trial46['auc']:>12.6f}"
        f"{trial46['ap']:>12.6f}"
    )

    print("\nDeltas Trial 46 vs Baseline:")

    print(
        f"  Ganancia : {fmt_millones(mejora_gain)} "
        f"({mejora_pct:+.2f}%)"
    )

    print(
        f"  AUC      : {delta_auc:+.6f}"
    )

    print(
        f"  AP       : {delta_ap:+.6f}"
    )

    # --------------------------------------------------------
    # Control contra resultado histórico z515
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONTROL DE REPRODUCIBILIDAD DEL BASELINE")
    print("=" * 70)

    print(
        f"Ganancia esperada : "
        f"{fmt_millones(BASELINE_REFERENCIA_GAIN)}"
    )

    print(
        f"Ganancia obtenida : "
        f"{fmt_millones(baseline['ganancia'])}"
    )

    print(
        f"Diferencia        : "
        f"{fmt_millones(
            baseline['ganancia']
            - BASELINE_REFERENCIA_GAIN
        )}"
    )

    print(
        f"\nAUC esperado      : "
        f"{BASELINE_REFERENCIA_AUC:.6f}"
    )

    print(
        f"AUC obtenido      : "
        f"{baseline['auc']:.6f}"
    )

    print(
        f"\nAP esperado       : "
        f"{BASELINE_REFERENCIA_AP:.6f}"
    )

    print(
        f"AP obtenido       : "
        f"{baseline['ap']:.6f}"
    )

    print(
        f"\nCorte esperado    : "
        f"{BASELINE_REFERENCIA_CORTE}"
    )

    print(
        f"Corte obtenido    : "
        f"{baseline['corte']}"
    )

    # --------------------------------------------------------
    # CSV resumen
    # --------------------------------------------------------

    resumen = pd.DataFrame(
        [
            {
                k: v
                for k, v in baseline.items()
                if k != "tabla"
            },
            {
                k: v
                for k, v in trial46.items()
                if k != "tabla"
            },
        ]
    )

    resumen_path = (
        OUTPUT_DIR / "comparacion_z518.csv"
    )

    resumen.to_csv(
        resumen_path,
        index=False,
    )

    print(
        f"\nResumen guardado en:\n"
        f"  {resumen_path}"
    )

    print("\nINTERPRETACION:")

    if mejora_gain > 0:
        print(
            "Trial 46 supera al baseline también "
            "en el período OOT 202105."
        )
        print(
            "Esto aporta evidencia temporal a favor "
            "de los hiperparámetros encontrados en z517."
        )
    else:
        print(
            "Trial 46 NO supera al baseline en "
            "el período OOT 202105."
        )
        print(
            "La mejora observada en 202106 podría "
            "estar parcialmente sobreajustada."
        )


if __name__ == "__main__":
    main()