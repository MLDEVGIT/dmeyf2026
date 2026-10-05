#!/usr/bin/env python3

"""
z535_competencia_01_aceleracion_screening.py

Competencia 01 - screening controlado de aceleraciones históricas.

Pregunta:
    ¿La segunda diferencia temporal de cada variable aporta señal
    incremental al modelo histórico FULL457?

Para cada variable x:

    delta_actual  = x_t - x_t-1
    delta_anterior = x_t-1 - x_t-2

    aceleracion =
        delta_actual - delta_anterior

    equivalente a:

        x_t - 2*x_t-1 + x_t-2

Diseño:
    Train : 202104 + 202105
    Test  : 202106
    Seed  : 290497
    Target: BAJA+2

Modelos:

    A - FULL457
        152 originales
        152 lag1
        152 delta_lag1
        lag1_disponible

    B - FULL_ACEL609
        FULL457
        + 152 aceleraciones

lag2_disponible:
    Se carga exclusivamente para auditoría.
    NO se utiliza como predictor.

IMPORTANTE:
    - Orden canónico: foto_mes, numero_de_cliente.
    - No usa 202108.
    - No genera submit.
    - No hace tuning.
    - No cambia hiperparámetros.
    - La única señal nueva en B son las 152 aceleraciones.
"""

from __future__ import annotations

import gc
import json
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
# CONFIGURACIÓN
# ============================================================

DATASET = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/feature_engineering_z532/"
    "competencia_01_historico_lag1_lag2.parquet"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/aceleracion_screening_z535"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"

LAG1_AVAILABLE_COL = "lag1_disponible"
LAG2_AVAILABLE_COL = "lag2_disponible"

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
EXPECTED_ACCELERATIONS = 152
EXPECTED_FULL_ACCEL = 609

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

def q(name: str) -> str:
    """Quote seguro de identificadores DuckDB."""
    return '"' + name.replace('"', '""') + '"'


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

    y_selected = y_true[selected]

    positives = int(
        y_selected.sum()
    )

    negatives = int(
        n - positives
    )

    gain = (
        positives * GAIN_POSITIVE
        + negatives * GAIN_NEGATIVE
    )

    precision = (
        positives / n
        if n > 0
        else np.nan
    )

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
        "gain_millions":
            gain / 1_000_000,
    }


def evaluate_model(
    name: str,
    feature_names: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[
    dict,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
]:

    print(
        "\n" + "=" * 80,
        flush=True,
    )

    print(
        f"MODELO: {name}",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Features: {len(feature_names)}",
        flush=True,
    )

    X_train = train_df[
        feature_names
    ]

    y_train = train_df[
        "target_binario"
    ].to_numpy(
        dtype=np.int8
    )

    X_test = test_df[
        feature_names
    ]

    y_test = test_df[
        "target_binario"
    ].to_numpy(
        dtype=np.int8
    )

    print(
        f"Train shape: {X_train.shape}",
        flush=True,
    )

    print(
        f"Test shape : {X_test.shape}",
        flush=True,
    )

    print(
        f"Train positivos: "
        f"{int(y_train.sum()):,}",
        flush=True,
    )

    print(
        f"Test positivos : "
        f"{int(y_test.sum()):,}",
        flush=True,
    )

    model = lgb.LGBMClassifier(
        **PARAMS
    )

    t0 = time.time()

    model.fit(
        X_train,
        y_train,
    )

    elapsed_train = (
        time.time() - t0
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    ap = average_precision_score(
        y_test,
        probabilities,
    )

    ll = log_loss(
        y_test,
        probabilities,
        labels=[0, 1],
    )

    rows = []

    for cut in DIAGNOSTIC_CUTS:
        rows.append(
            gain_at_n(
                y_test,
                probabilities,
                cut,
            )
        )

    gain_curve = pd.DataFrame(
        rows
    )

    best_row = (
        gain_curve
        .sort_values(
            [
                "gain",
                "cut",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    primary_row = (
        gain_curve[
            gain_curve["cut"]
            == PRIMARY_CUT
        ]
        .iloc[0]
    )

    importance = pd.DataFrame(
        {
            "feature":
                feature_names,

            "importance_gain":
                model.booster_
                .feature_importance(
                    importance_type="gain"
                ),

            "importance_split":
                model.booster_
                .feature_importance(
                    importance_type="split"
                ),
        }
    )

    importance = (
        importance
        .sort_values(
            "importance_gain",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    summary = {
        "model": name,
        "features":
            len(feature_names),
        "auc":
            float(auc),
        "average_precision":
            float(ap),
        "log_loss":
            float(ll),
        "training_seconds":
            float(elapsed_train),
        "primary_cut":
            PRIMARY_CUT,
        "primary_positives":
            int(
                primary_row[
                    "positives"
                ]
            ),
        "primary_gain":
            int(
                primary_row[
                    "gain"
                ]
            ),
        "primary_gain_millions":
            float(
                primary_row[
                    "gain_millions"
                ]
            ),
        "best_cut":
            int(
                best_row[
                    "cut"
                ]
            ),
        "best_positives":
            int(
                best_row[
                    "positives"
                ]
            ),
        "best_gain":
            int(
                best_row[
                    "gain"
                ]
            ),
        "best_gain_millions":
            float(
                best_row[
                    "gain_millions"
                ]
            ),
    }

    print(
        f"AUC      : {auc:.6f}",
        flush=True,
    )

    print(
        f"AP       : {ap:.6f}",
        flush=True,
    )

    print(
        f"LogLoss  : {ll:.6f}",
        flush=True,
    )

    print(
        f"Tiempo   : "
        f"{elapsed_train:.1f}s",
        flush=True,
    )

    print(
        f"N={PRIMARY_CUT:,}: "
        f"positivos="
        f"{int(primary_row['positives']):,} "
        f"| gain="
        f"{primary_row['gain_millions']:.3f} M",
        flush=True,
    )

    print(
        f"Mejor diagnóstico: "
        f"N={int(best_row['cut']):,} "
        f"| positivos="
        f"{int(best_row['positives']):,} "
        f"| gain="
        f"{best_row['gain_millions']:.3f} M",
        flush=True,
    )

    del X_train
    del X_test
    del model

    gc.collect()

    return (
        summary,
        gain_curve,
        importance,
        probabilities,
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
        "z535 - SCREENING DE ACELERACIONES HISTÓRICAS",
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

    # ========================================================
    # LEER ESQUEMA
    # ========================================================

    con = duckdb.connect()

    schema_df = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet(
            '{DATASET}'
        )
        """
    ).df()

    all_columns = (
        schema_df[
            "column_name"
        ]
        .tolist()
    )

    metadata = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
        LAG1_AVAILABLE_COL,
        LAG2_AVAILABLE_COL,
    }

    originales = [
        c
        for c in all_columns
        if c not in metadata
        and not c.endswith(
            "_lag1"
        )
        and not c.endswith(
            "_delta_lag1"
        )
        and not c.endswith(
            "_lag2"
        )
    ]

    lag1 = [
        f"{c}_lag1"
        for c in originales
    ]

    delta1 = [
        f"{c}_delta_lag1"
        for c in originales
    ]

    lag2 = [
        f"{c}_lag2"
        for c in originales
    ]

    accel_features = [
        f"{c}_acel"
        for c in originales
    ]

    # ========================================================
    # VALIDAR ESQUEMA
    # ========================================================

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

    for cols, label in [
        (lag1, "lag1"),
        (delta1, "delta_lag1"),
        (lag2, "lag2"),
    ]:

        missing = [
            c
            for c in cols
            if c not in all_columns
        ]

        if missing:
            raise ValueError(
                f"Faltan columnas {label}: "
                + ", ".join(
                    missing[:10]
                )
            )

    full_features = (
        originales
        + lag1
        + delta1
        + [LAG1_AVAILABLE_COL]
    )

    full_accel_features = (
        full_features
        + accel_features
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

    if (
        len(accel_features)
        != EXPECTED_ACCELERATIONS
    ):
        raise ValueError(
            f"Debería haber "
            f"{EXPECTED_ACCELERATIONS} "
            f"aceleraciones y hay "
            f"{len(accel_features)}."
        )

    if (
        len(full_accel_features)
        != EXPECTED_FULL_ACCEL
    ):
        raise ValueError(
            f"FULL+ACEL debería tener "
            f"{EXPECTED_FULL_ACCEL} "
            f"features y tiene "
            f"{len(full_accel_features)}."
        )

    print(
        "\nFeatures:",
        flush=True,
    )

    print(
        f"  Originales      : "
        f"{len(originales)}",
        flush=True,
    )

    print(
        f"  Lag1            : "
        f"{len(lag1)}",
        flush=True,
    )

    print(
        f"  Delta lag1      : "
        f"{len(delta1)}",
        flush=True,
    )

    print(
        f"  FULL            : "
        f"{len(full_features)}",
        flush=True,
    )

    print(
        f"  Aceleraciones   : "
        f"{len(accel_features)}",
        flush=True,
    )

    print(
        f"  FULL + ACEL     : "
        f"{len(full_accel_features)}",
        flush=True,
    )

    print(
        f"  {LAG2_AVAILABLE_COL}: "
        f"solo auditoría",
        flush=True,
    )

    # ========================================================
    # SELECT SQL
    # ========================================================

    base_select_cols = (
        [
            ID_COL,
            MONTH_COL,
            TARGET_COL,
            LAG1_AVAILABLE_COL,
            LAG2_AVAILABLE_COL,
        ]
        + originales
        + lag1
        + delta1
        + lag2
    )

    duplicated = [
        c
        for c in set(
            base_select_cols
        )
        if base_select_cols.count(c) > 1
    ]

    if duplicated:
        raise ValueError(
            "Columnas duplicadas: "
            + ", ".join(
                sorted(duplicated)
            )
        )

    select_parts = [
        q(c)
        for c in base_select_cols
    ]

    # Segunda diferencia:
    #
    # x_t - 2*x_t-1 + x_t-2
    #
    # CAST a DOUBLE para evitar problemas de tipo
    # con las variables enteras.
    for c in originales:

        expression = (
            f"CAST({q(c)} AS DOUBLE)"
            f" - 2.0 * "
            f"CAST({q(c + '_lag1')} AS DOUBLE)"
            f" + "
            f"CAST({q(c + '_lag2')} AS DOUBLE)"
            f" AS {q(c + '_acel')}"
        )

        select_parts.append(
            expression
        )

    select_sql = ",\n".join(
        select_parts
    )

    months_sql = ", ".join(
        str(x)
        for x in (
            TRAIN_MONTHS
            + [TEST_MONTH]
        )
    )

    # ========================================================
    # CARGA CANÓNICA
    # ========================================================

    print(
        "\nCargando train/test "
        "y calculando aceleraciones...",
        flush=True,
    )

    t_load = time.time()

    df = con.execute(
        f"""
        SELECT
            {select_sql}
        FROM read_parquet(
            '{DATASET}'
        )
        WHERE
            {q(MONTH_COL)}
            IN ({months_sql})
        ORDER BY
            {q(MONTH_COL)},
            {q(ID_COL)}
        """
    ).df()

    con.close()

    print(
        f"Carga completada en "
        f"{time.time() - t_load:.1f}s",
        flush=True,
    )

    print(
        f"Filas: {len(df):,}",
        flush=True,
    )

    # ========================================================
    # TARGET
    # ========================================================

    df["target_binario"] = (
        df[TARGET_COL]
        == "BAJA+2"
    ).astype(
        np.int8
    )

    train_df = (
        df[
            df[MONTH_COL]
            .isin(
                TRAIN_MONTHS
            )
        ]
        .reset_index(
            drop=True
        )
    )

    test_df = (
        df[
            df[MONTH_COL]
            == TEST_MONTH
        ]
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nTrain rows: "
        f"{len(train_df):,}",
        flush=True,
    )

    print(
        f"Test rows : "
        f"{len(test_df):,}",
        flush=True,
    )

    print(
        f"Train BAJA+2: "
        f"{int(train_df['target_binario'].sum()):,}",
        flush=True,
    )

    print(
        f"Test BAJA+2 : "
        f"{int(test_df['target_binario'].sum()):,}",
        flush=True,
    )

    print(
        "Primeros meses train:",
        train_df[
            MONTH_COL
        ].head(10).tolist(),
        flush=True,
    )

    # ========================================================
    # AUDITORÍA LAG2 / ACELERACIÓN
    # ========================================================

    print(
        "\n" + "=" * 80,
        flush=True,
    )

    print(
        "DISPONIBILIDAD HISTÓRICA",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    availability = (
        df.groupby(
            MONTH_COL
        )[
            LAG2_AVAILABLE_COL
        ]
        .agg(
            [
                "count",
                "sum",
                "mean",
            ]
        )
        .reset_index()
    )

    for _, row in (
        availability.iterrows()
    ):

        print(
            f"{int(row[MONTH_COL])} "
            f"| filas="
            f"{int(row['count']):,} "
            f"| con_lag2="
            f"{int(row['sum']):,} "
            f"| cobertura="
            f"{100 * row['mean']:.4f}%",
            flush=True,
        )

    availability.to_csv(
        OUTPUT_DIR
        / "lag2_availability.csv",
        index=False,
    )

    # Auditoría real de las aceleraciones.
    accel_non_null = (
        df.groupby(
            MONTH_COL
        )[accel_features]
        .apply(
            lambda x:
                x.notna()
                .any(axis=1)
                .sum()
        )
    )

    print(
        "\nFilas con al menos una "
        "aceleración disponible:",
        flush=True,
    )

    for month, count in (
        accel_non_null.items()
    ):

        total = int(
            (
                df[MONTH_COL]
                == month
            ).sum()
        )

        print(
            f"{int(month)} "
            f"| {int(count):,}/{total:,} "
            f"| "
            f"{100 * count / total:.4f}%",
            flush=True,
        )

    # ========================================================
    # A - FULL457
    # ========================================================

    (
        summary_full,
        curve_full,
        imp_full,
        prob_full,
    ) = evaluate_model(
        name="FULL457",
        feature_names=
            full_features,
        train_df=
            train_df,
        test_df=
            test_df,
    )

    curve_full.to_csv(
        OUTPUT_DIR
        / "gain_curve_FULL457.csv",
        index=False,
    )

    imp_full.to_csv(
        OUTPUT_DIR
        / "importance_FULL457.csv",
        index=False,
    )

    # ========================================================
    # B - FULL + ACELERACIONES
    # ========================================================

    (
        summary_accel,
        curve_accel,
        imp_accel,
        prob_accel,
    ) = evaluate_model(
        name="FULL_ACEL609",
        feature_names=
            full_accel_features,
        train_df=
            train_df,
        test_df=
            test_df,
    )

    curve_accel.to_csv(
        OUTPUT_DIR
        / "gain_curve_FULL_ACEL609.csv",
        index=False,
    )

    imp_accel.to_csv(
        OUTPUT_DIR
        / "importance_FULL_ACEL609.csv",
        index=False,
    )

    # ========================================================
    # COMPARACIÓN
    # ========================================================

    comparison = (
        curve_full[
            [
                "cut",
                "positives",
                "gain_millions",
            ]
        ]
        .rename(
            columns={
                "positives":
                    "positives_FULL",
                "gain_millions":
                    "gain_FULL_millions",
            }
        )
        .merge(
            curve_accel[
                [
                    "cut",
                    "positives",
                    "gain_millions",
                ]
            ].rename(
                columns={
                    "positives":
                        "positives_ACEL",
                    "gain_millions":
                        "gain_ACEL_millions",
                }
            ),
            on="cut",
        )
    )

    comparison[
        "delta_ACEL_vs_FULL_millions"
    ] = (
        comparison[
            "gain_ACEL_millions"
        ]
        - comparison[
            "gain_FULL_millions"
        ]
    )

    print(
        "\n" + "=" * 80,
        flush=True,
    )

    print(
        "COMPARACIÓN",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        comparison.to_string(
            index=False
        ),
        flush=True,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "comparison_by_cut.csv",
        index=False,
    )

    # ========================================================
    # DIVERSIDAD DE RANKING
    # ========================================================

    correlation = float(
        np.corrcoef(
            prob_full,
            prob_accel,
        )[0, 1]
    )

    max_abs_diff = float(
        np.max(
            np.abs(
                prob_full
                - prob_accel
            )
        )
    )

    order_full = np.argsort(
        -prob_full,
        kind="stable",
    )

    order_accel = np.argsort(
        -prob_accel,
        kind="stable",
    )

    top_full = set(
        test_df.iloc[
            order_full[
                :PRIMARY_CUT
            ]
        ][ID_COL]
        .astype(int)
        .tolist()
    )

    top_accel = set(
        test_df.iloc[
            order_accel[
                :PRIMARY_CUT
            ]
        ][ID_COL]
        .astype(int)
        .tolist()
    )

    jaccard_top = (
        len(
            top_full
            & top_accel
        )
        / len(
            top_full
            | top_accel
        )
    )

    print(
        "\nDiversidad de ranking:",
        flush=True,
    )

    print(
        f"Correlación probabilidades: "
        f"{correlation:.9f}",
        flush=True,
    )

    print(
        f"Máxima diferencia absoluta: "
        f"{max_abs_diff:.9f}",
        flush=True,
    )

    print(
        f"Jaccard top {PRIMARY_CUT:,}: "
        f"{jaccard_top:.6f}",
        flush=True,
    )

    # ========================================================
    # IMPORTANCIA DE ACELERACIONES
    # ========================================================

    accel_importance = (
        imp_accel[
            imp_accel[
                "feature"
            ].isin(
                accel_features
            )
        ]
        .copy()
    )

    total_gain_importance = (
        imp_accel[
            "importance_gain"
        ].sum()
    )

    accel_gain_importance = (
        accel_importance[
            "importance_gain"
        ].sum()
    )

    accel_share = (
        accel_gain_importance
        / total_gain_importance
        if total_gain_importance > 0
        else np.nan
    )

    accel_importance.to_csv(
        OUTPUT_DIR
        / "importance_accelerations.csv",
        index=False,
    )

    print(
        f"\nShare gain importance "
        f"aceleraciones: "
        f"{100 * accel_share:.2f}%",
        flush=True,
    )

    print(
        "\nTop 20 aceleraciones:",
        flush=True,
    )

    print(
        accel_importance[
            [
                "feature",
                "importance_gain",
                "importance_split",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        ),
        flush=True,
    )

    # ========================================================
    # RESUMEN / AUDITORÍA
    # ========================================================

    summaries = pd.DataFrame(
        [
            summary_full,
            summary_accel,
        ]
    )

    summaries.to_csv(
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

        "ordering":
            [
                MONTH_COL,
                ID_COL,
            ],

        "target":
            "BAJA+2",

        "original_features":
            len(originales),

        "full_features":
            len(full_features),

        "acceleration_features":
            len(accel_features),

        "full_acceleration_features":
            len(
                full_accel_features
            ),

        "acceleration_formula":
            "x_t - 2*x_t-1 + x_t-2",

        "train_rows":
            len(train_df),

        "test_rows":
            len(test_df),

        "train_positives":
            int(
                train_df[
                    "target_binario"
                ].sum()
            ),

        "test_positives":
            int(
                test_df[
                    "target_binario"
                ].sum()
            ),

        "probability_correlation":
            correlation,

        "max_abs_probability_difference":
            max_abs_diff,

        "jaccard_top_primary_cut":
            jaccard_top,

        "acceleration_gain_importance_share":
            float(accel_share),

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

    # ========================================================
    # RESULTADO PRINCIPAL
    # ========================================================

    full_gain = (
        summary_full[
            "primary_gain_millions"
        ]
    )

    accel_gain = (
        summary_accel[
            "primary_gain_millions"
        ]
    )

    delta_gain = (
        accel_gain
        - full_gain
    )

    delta_pct = (
        100 * delta_gain / full_gain
        if full_gain != 0
        else np.nan
    )

    print(
        "\n" + "=" * 80,
        flush=True,
    )

    print(
        "RESULTADO PRINCIPAL",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"FULL457 N={PRIMARY_CUT:,}: "
        f"{full_gain:.3f} M",
        flush=True,
    )

    print(
        f"FULL+ACEL N={PRIMARY_CUT:,}: "
        f"{accel_gain:.3f} M",
        flush=True,
    )

    print(
        f"Delta: "
        f"{delta_gain:+.3f} M "
        f"({delta_pct:+.2f}%)",
        flush=True,
    )

    print(
        "\nArchivos guardados en:",
        OUTPUT_DIR,
        flush=True,
    )

    print(
        "\nz535 finalizado.",
        flush=True,
    )


if __name__ == "__main__":
    main()
