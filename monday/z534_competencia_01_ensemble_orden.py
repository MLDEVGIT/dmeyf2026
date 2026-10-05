#!/usr/bin/env python3

"""
z534_competencia_01_ensemble_orden.py

Competencia 01 - screening controlado del efecto del orden
de las filas y de un ensemble entre órdenes.

Pregunta:
    Dado que LightGBM produjo rankings distintos con exactamente
    los mismos datos, features, hiperparámetros y seed al cambiar
    el orden de las filas de entrenamiento:

    ¿el promedio de probabilidades entre ambos modelos aporta
    estabilidad o mejora de ganancia?

Diseño:
    Train : 202104 + 202105
    Test  : 202106
    Seed  : 290497
    Target: BAJA+2

Features:
    FULL457
        152 originales
        152 lag1
        152 delta_lag1
        lag1_disponible

Modelos:
    A - NATURAL
        Conserva el orden físico del parquet z523.

    B - MES_CLIENTE
        Ordena train por foto_mes, numero_de_cliente.

    C - ENSEMBLE_50_50
        Promedio simple de probabilidades A y B.

IMPORTANTE:
    - A y B usan exactamente los mismos registros.
    - A y B usan exactamente las mismas 457 features.
    - A y B usan exactamente los mismos hiperparámetros.
    - A y B usan exactamente la misma seed.
    - La única diferencia es el orden de las filas del train.
    - Test conserva el mismo orden para ambos modelos.
    - No usa 202108.
    - No genera submit.
    - No hace tuning.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/feature_engineering_z523/"
    "competencia_01_historico_lag1.parquet"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/ensemble_orden_z534"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
LAG1_AVAILABLE_COL = "lag1_disponible"

TRAIN_MONTHS = [202104, 202105]
TEST_MONTH = 202106

SEED = 290497

PRIMARY_CUT = 12000

DIAGNOSTIC_CUTS = list(
    range(8000, 16001, 500)
)

GAIN_POSITIVE = 1_072_500
GAIN_NEGATIVE = -27_500

EXPECTED_ORIGINALS = 152
EXPECTED_FULL = 457

PARAMS = {
    "objective": "binary",
    "n_estimators": 1200,
    "learning_rate": 0.02,
    "num_leaves": 750,
    "max_depth": -1,
    "min_child_samples": 5000,
    "max_bin": 31,
    "colsample_bytree": 0.5,
    "subsample": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}


# ============================================================
# HELPERS
# ============================================================

def gain_at_n(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n: int,
) -> dict:

    n = min(n, len(y_true))

    order = np.argsort(
        -probabilities,
        kind="stable",
    )

    selected = order[:n]

    positives = int(
        y_true[selected].sum()
    )

    negatives = int(
        n - positives
    )

    gain = (
        positives * GAIN_POSITIVE
        + negatives * GAIN_NEGATIVE
    )

    precision = positives / n

    total_positives = int(
        y_true.sum()
    )

    recall = (
        positives / total_positives
        if total_positives > 0
        else np.nan
    )

    return {
        "cut": int(n),
        "positives": positives,
        "negatives": negatives,
        "precision": precision,
        "recall": recall,
        "gain": int(gain),
        "gain_millions": gain / 1_000_000,
    }


def evaluate_probabilities(
    name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict, pd.DataFrame]:

    auc = roc_auc_score(
        y_true,
        probabilities,
    )

    ap = average_precision_score(
        y_true,
        probabilities,
    )

    ll = log_loss(
        y_true,
        probabilities,
        labels=[0, 1],
    )

    rows = [
        gain_at_n(
            y_true,
            probabilities,
            cut,
        )
        for cut in DIAGNOSTIC_CUTS
    ]

    curve = pd.DataFrame(rows)

    primary = (
        curve[
            curve["cut"] == PRIMARY_CUT
        ]
        .iloc[0]
    )

    best = (
        curve
        .sort_values(
            ["gain", "cut"],
            ascending=[False, True],
        )
        .iloc[0]
    )

    summary = {
        "model": name,
        "auc": float(auc),
        "average_precision": float(ap),
        "log_loss": float(ll),
        "primary_cut": PRIMARY_CUT,
        "primary_positives":
            int(primary["positives"]),
        "primary_gain":
            int(primary["gain"]),
        "primary_gain_millions":
            float(primary["gain_millions"]),
        "best_cut":
            int(best["cut"]),
        "best_positives":
            int(best["positives"]),
        "best_gain":
            int(best["gain"]),
        "best_gain_millions":
            float(best["gain_millions"]),
    }

    print(
        f"\n{name}",
        flush=True,
    )

    print(
        f"  AUC      : {auc:.6f}",
        flush=True,
    )

    print(
        f"  AP       : {ap:.6f}",
        flush=True,
    )

    print(
        f"  LogLoss  : {ll:.6f}",
        flush=True,
    )

    print(
        f"  N={PRIMARY_CUT:,}: "
        f"positivos={int(primary['positives']):,} "
        f"| gain={primary['gain_millions']:.3f} M",
        flush=True,
    )

    print(
        f"  Mejor diagnóstico: "
        f"N={int(best['cut']):,} "
        f"| positivos={int(best['positives']):,} "
        f"| gain={best['gain_millions']:.3f} M",
        flush=True,
    )

    return summary, curve


def train_model(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, float]:

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        f"ENTRENANDO: {name}",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "Primeros meses train:",
        train_df[MONTH_COL]
        .head(10)
        .tolist(),
        flush=True,
    )

    print(
        f"Train rows     : {len(train_df):,}",
        flush=True,
    )

    print(
        f"Train positivos: "
        f"{int(train_df['target_binario'].sum()):,}",
        flush=True,
    )

    model = lgb.LGBMClassifier(
        **PARAMS
    )

    t0 = time.time()

    model.fit(
        train_df[features],
        train_df[
            "target_binario"
        ].to_numpy(
            dtype=np.int8
        ),
    )

    elapsed = time.time() - t0

    probabilities = (
        model.predict_proba(
            test_df[features]
        )[:, 1]
    )

    print(
        f"Tiempo entrenamiento: "
        f"{elapsed:.1f}s",
        flush=True,
    )

    return probabilities, elapsed


def top_n_ids(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    n: int,
) -> set[int]:

    order = np.argsort(
        -probabilities,
        kind="stable",
    )

    idx = order[:n]

    return set(
        test_df.iloc[idx][ID_COL]
        .astype(int)
        .tolist()
    )


def jaccard(
    a: set[int],
    b: set[int],
) -> float:

    return (
        len(a & b)
        / len(a | b)
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "z534 - ENSEMBLE POR ORDEN DE FILAS",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Dataset: {DATASET}",
        flush=True,
    )

    # --------------------------------------------------------
    # CARGA
    # --------------------------------------------------------

    print(
        "\nCargando parquet z523...",
        flush=True,
    )

    t0 = time.time()

    df = pd.read_parquet(
        DATASET
    )

    df = (
        df[
            df[MONTH_COL]
            .isin(
                TRAIN_MONTHS
                + [TEST_MONTH]
            )
        ]
        .copy()
    )

    print(
        f"Carga completada en "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )

    print(
        f"Filas cargadas: {len(df):,}",
        flush=True,
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    metadata = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
        LAG1_AVAILABLE_COL,
    }

    delta1 = [
        c
        for c in df.columns
        if c.endswith(
            "_delta_lag1"
        )
    ]

    lag1 = [
        c
        for c in df.columns
        if c.endswith("_lag1")
        and not c.endswith(
            "_delta_lag1"
        )
    ]

    originales = [
        c
        for c in df.columns
        if c not in metadata
        and not c.endswith(
            "_lag1"
        )
        and not c.endswith(
            "_delta_lag1"
        )
    ]

    full_features = (
        originales
        + lag1
        + delta1
        + [LAG1_AVAILABLE_COL]
    )

    if (
        len(originales)
        != EXPECTED_ORIGINALS
    ):
        raise ValueError(
            f"Esperaba "
            f"{EXPECTED_ORIGINALS} originales "
            f"y encontré "
            f"{len(originales)}."
        )

    if (
        len(full_features)
        != EXPECTED_FULL
    ):
        raise ValueError(
            f"FULL debería tener "
            f"{EXPECTED_FULL} features "
            f"y tiene "
            f"{len(full_features)}."
        )

    print(
        "\nFeatures:",
        flush=True,
    )

    print(
        f"  Originales : "
        f"{len(originales)}",
        flush=True,
    )

    print(
        f"  Lag1       : "
        f"{len(lag1)}",
        flush=True,
    )

    print(
        f"  Delta lag1 : "
        f"{len(delta1)}",
        flush=True,
    )

    print(
        f"  FULL       : "
        f"{len(full_features)}",
        flush=True,
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    df["target_binario"] = (
        df[TARGET_COL]
        == "BAJA+2"
    ).astype(
        np.int8
    )

    # --------------------------------------------------------
    # TEST ÚNICO
    # --------------------------------------------------------

    test_df = (
        df[
            df[MONTH_COL]
            == TEST_MONTH
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # Ordenamos SOLO el test para tener una referencia única.
    # Esto no cambia el experimento porque ambos modelos
    # predicen exactamente sobre este mismo dataframe.
    test_df = (
        test_df
        .sort_values(
            [MONTH_COL, ID_COL],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    y_test = (
        test_df[
            "target_binario"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )

    # --------------------------------------------------------
    # TRAIN A - ORDEN NATURAL
    # --------------------------------------------------------

    train_natural = (
        df[
            df[MONTH_COL]
            .isin(
                TRAIN_MONTHS
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # TRAIN B - ORDEN CANÓNICO MES + CLIENTE
    # --------------------------------------------------------

    train_sorted = (
        train_natural
        .sort_values(
            [MONTH_COL, ID_COL],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # VALIDACIÓN: MISMO CONTENIDO
    # --------------------------------------------------------

    natural_keys = (
        train_natural[
            [MONTH_COL, ID_COL]
        ]
        .sort_values(
            [MONTH_COL, ID_COL],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    sorted_keys = (
        train_sorted[
            [MONTH_COL, ID_COL]
        ]
        .reset_index(
            drop=True
        )
    )

    if not natural_keys.equals(
        sorted_keys
    ):
        raise ValueError(
            "Los dos trains no contienen "
            "exactamente los mismos registros."
        )

    print(
        "\nTrain:",
        flush=True,
    )

    print(
        f"  filas     : "
        f"{len(train_natural):,}",
        flush=True,
    )

    print(
        f"  positivos : "
        f"{int(train_natural['target_binario'].sum()):,}",
        flush=True,
    )

    print(
        "  orden natural primeros meses:",
        train_natural[
            MONTH_COL
        ].head(10).tolist(),
        flush=True,
    )

    print(
        "  orden sorted primeros meses :",
        train_sorted[
            MONTH_COL
        ].head(10).tolist(),
        flush=True,
    )

    # --------------------------------------------------------
    # MODELO A
    # --------------------------------------------------------

    prob_natural, time_natural = (
        train_model(
            name="A_NATURAL",
            train_df=train_natural,
            test_df=test_df,
            features=full_features,
        )
    )

    # --------------------------------------------------------
    # MODELO B
    # --------------------------------------------------------

    prob_sorted, time_sorted = (
        train_model(
            name="B_MES_CLIENTE",
            train_df=train_sorted,
            test_df=test_df,
            features=full_features,
        )
    )

    # --------------------------------------------------------
    # ENSEMBLE
    # --------------------------------------------------------

    prob_ensemble = (
        prob_natural
        + prob_sorted
    ) / 2.0

    # --------------------------------------------------------
    # EVALUACIÓN
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "RESULTADOS",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    summary_a, curve_a = (
        evaluate_probabilities(
            "A_NATURAL",
            y_test,
            prob_natural,
        )
    )

    summary_b, curve_b = (
        evaluate_probabilities(
            "B_MES_CLIENTE",
            y_test,
            prob_sorted,
        )
    )

    summary_c, curve_c = (
        evaluate_probabilities(
            "C_ENSEMBLE_50_50",
            y_test,
            prob_ensemble,
        )
    )

    # --------------------------------------------------------
    # DIVERSIDAD
    # --------------------------------------------------------

    top_a = top_n_ids(
        test_df,
        prob_natural,
        PRIMARY_CUT,
    )

    top_b = top_n_ids(
        test_df,
        prob_sorted,
        PRIMARY_CUT,
    )

    top_c = top_n_ids(
        test_df,
        prob_ensemble,
        PRIMARY_CUT,
    )

    jac_ab = jaccard(
        top_a,
        top_b,
    )

    jac_ac = jaccard(
        top_a,
        top_c,
    )

    jac_bc = jaccard(
        top_b,
        top_c,
    )

    corr_ab = float(
        np.corrcoef(
            prob_natural,
            prob_sorted,
        )[0, 1]
    )

    max_abs_diff = float(
        np.max(
            np.abs(
                prob_natural
                - prob_sorted
            )
        )
    )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "DIVERSIDAD",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Correlación probs A/B : "
        f"{corr_ab:.9f}",
        flush=True,
    )

    print(
        f"Máx diferencia probs  : "
        f"{max_abs_diff:.9f}",
        flush=True,
    )

    print(
        f"Jaccard A/B top12k    : "
        f"{jac_ab:.6f}",
        flush=True,
    )

    print(
        f"Jaccard A/C top12k    : "
        f"{jac_ac:.6f}",
        flush=True,
    )

    print(
        f"Jaccard B/C top12k    : "
        f"{jac_bc:.6f}",
        flush=True,
    )

    # --------------------------------------------------------
    # COMPARACIÓN POR CUT
    # --------------------------------------------------------

    comparison = (
        curve_a[
            [
                "cut",
                "positives",
                "gain_millions",
            ]
        ]
        .rename(
            columns={
                "positives":
                    "positives_A",
                "gain_millions":
                    "gain_A_millions",
            }
        )
        .merge(
            curve_b[
                [
                    "cut",
                    "positives",
                    "gain_millions",
                ]
            ].rename(
                columns={
                    "positives":
                        "positives_B",
                    "gain_millions":
                        "gain_B_millions",
                }
            ),
            on="cut",
        )
        .merge(
            curve_c[
                [
                    "cut",
                    "positives",
                    "gain_millions",
                ]
            ].rename(
                columns={
                    "positives":
                        "positives_C",
                    "gain_millions":
                        "gain_C_millions",
                }
            ),
            on="cut",
        )
    )

    comparison[
        "delta_C_vs_A_millions"
    ] = (
        comparison[
            "gain_C_millions"
        ]
        - comparison[
            "gain_A_millions"
        ]
    )

    comparison[
        "delta_C_vs_B_millions"
    ] = (
        comparison[
            "gain_C_millions"
        ]
        - comparison[
            "gain_B_millions"
        ]
    )

    print(
        "\nComparación por corte:",
        flush=True,
    )

    print(
        comparison.to_string(
            index=False
        ),
        flush=True,
    )

    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    curve_a.to_csv(
        OUTPUT_DIR
        / "gain_curve_A_natural.csv",
        index=False,
    )

    curve_b.to_csv(
        OUTPUT_DIR
        / "gain_curve_B_mes_cliente.csv",
        index=False,
    )

    curve_c.to_csv(
        OUTPUT_DIR
        / "gain_curve_C_ensemble.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "comparison_by_cut.csv",
        index=False,
    )

    predictions = pd.DataFrame(
        {
            ID_COL:
                test_df[
                    ID_COL
                ].astype(int),

            "target_binario":
                y_test,

            "prob_natural":
                prob_natural,

            "prob_mes_cliente":
                prob_sorted,

            "prob_ensemble":
                prob_ensemble,
        }
    )

    predictions.to_parquet(
        OUTPUT_DIR
        / "predictions.parquet",
        index=False,
    )

    summaries = [
        summary_a,
        summary_b,
        summary_c,
    ]

    pd.DataFrame(
        summaries
    ).to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    audit = {
        "dataset":
            str(DATASET),

        "train_months":
            TRAIN_MONTHS,

        "test_month":
            TEST_MONTH,

        "seed":
            SEED,

        "features":
            len(full_features),

        "train_rows":
            len(train_natural),

        "test_rows":
            len(test_df),

        "train_positives":
            int(
                train_natural[
                    "target_binario"
                ].sum()
            ),

        "test_positives":
            int(
                y_test.sum()
            ),

        "correlation_A_B":
            corr_ab,

        "max_abs_probability_difference":
            max_abs_diff,

        "jaccard_A_B_top12000":
            jac_ab,

        "jaccard_A_C_top12000":
            jac_ac,

        "jaccard_B_C_top12000":
            jac_bc,

        "training_seconds_A":
            time_natural,

        "training_seconds_B":
            time_sorted,

        "params":
            PARAMS,
    }

    with open(
        OUTPUT_DIR
        / "audit.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
        )

    print(
        "\nArchivos guardados en:",
        OUTPUT_DIR,
        flush=True,
    )

    print(
        "\nz534 finalizado.",
        flush=True,
    )


if __name__ == "__main__":
    main()
