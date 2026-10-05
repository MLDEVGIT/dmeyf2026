#!/usr/bin/env python3

"""
z533_competencia_01_lag2_screening.py

Screening controlado de lag2 para Competencia 01.

Pregunta:
    ¿Agregar las 152 variables lag2 al modelo histórico FULL actual
    aporta señal incremental?

Diseño:
    Train : 202104 + 202105
    Test  : 202106
    Seed  : 290497

Modelos:
    FULL457
        152 originales
        152 lag1
        152 delta_lag1
        lag1_disponible

    FULL_LAG2_609
        FULL457
        + 152 lag2

lag2_disponible:
    Se carga exclusivamente para auditoría.
    NO se utiliza como predictor.

IMPORTANTE:
    - No usa 202108.
    - No genera submit.
    - No hace tuning.
    - No cambia hiperparámetros.
    - La única diferencia entre A y B es lag2.
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
    "competencia_01/lag2_screening_z533"
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
EXPECTED_FULL_LAG2 = 609

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
    """
    Calcula ganancia económica seleccionando los N clientes
    con mayor probabilidad estimada.
    """

    n = min(
        n,
        len(y_true),
    )

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
        "gain_millions": gain / 1_000_000,
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
]:
    """
    Entrena un LightGBM y devuelve:
        - resumen de métricas
        - curva de ganancia
        - feature importance
    """

    print(
        "\n"
        + "=" * 80,
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
        time.time()
        - t0
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
        "features": len(feature_names),
        "seed": SEED,
        "train_months":
            ",".join(
                map(
                    str,
                    TRAIN_MONTHS,
                )
            ),
        "test_month":
            TEST_MONTH,
        "train_rows":
            len(train_df),
        "test_rows":
            len(test_df),
        "train_positives":
            int(y_train.sum()),
        "test_positives":
            int(y_test.sum()),
        "auc":
            float(auc),
        "average_precision":
            float(ap),
        "log_loss":
            float(ll),
        "gain_n12000":
            int(
                primary_row["gain"]
            ),
        "gain_n12000_millions":
            float(
                primary_row[
                    "gain_millions"
                ]
            ),
        "positives_n12000":
            int(
                primary_row[
                    "positives"
                ]
            ),
        "precision_n12000":
            float(
                primary_row[
                    "precision"
                ]
            ),
        "recall_n12000":
            float(
                primary_row[
                    "recall"
                ]
            ),
        "best_cut":
            int(
                best_row["cut"]
            ),
        "best_gain":
            int(
                best_row["gain"]
            ),
        "best_gain_millions":
            float(
                best_row[
                    "gain_millions"
                ]
            ),
        "best_positives":
            int(
                best_row[
                    "positives"
                ]
            ),
        "train_seconds":
            elapsed_train,
    }

    print(
        f"\nAUC              : "
        f"{auc:.6f}",
        flush=True,
    )

    print(
        f"Average Precision: "
        f"{ap:.6f}",
        flush=True,
    )

    print(
        f"Log Loss         : "
        f"{ll:.6f}",
        flush=True,
    )

    print(
        f"\nN={PRIMARY_CUT:,}",
        flush=True,
    )

    print(
        f"Positivos        : "
        f"{int(primary_row['positives']):,}",
        flush=True,
    )

    print(
        f"Precision        : "
        f"{primary_row['precision']:.4f}",
        flush=True,
    )

    print(
        f"Recall           : "
        f"{primary_row['recall']:.4f}",
        flush=True,
    )

    print(
        f"Ganancia         : "
        f"{primary_row['gain_millions']:.3f} M",
        flush=True,
    )

    print(
        f"\nMejor corte diagnóstico: "
        f"{int(best_row['cut']):,}",
        flush=True,
    )

    print(
        f"Mejor ganancia          : "
        f"{best_row['gain_millions']:.3f} M",
        flush=True,
    )

    print(
        f"\nTiempo entrenamiento: "
        f"{elapsed_train:.1f}s",
        flush=True,
    )

    del model
    del X_train
    del X_test

    gc.collect()

    return (
        summary,
        gain_curve,
        importance,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    start = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "z533 - SCREENING CONTROLADO LAG2",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"\nDataset:\n  {DATASET}",
        flush=True,
    )

    print(
        f"\nTrain: {TRAIN_MONTHS}",
        flush=True,
    )

    print(
        f"Test : {TEST_MONTH}",
        flush=True,
    )

    print(
        f"Seed : {SEED}",
        flush=True,
    )

    print(
        f"Corte principal: "
        f"{PRIMARY_CUT:,}",
        flush=True,
    )

    print(
        "\nNO usa 202108.",
        flush=True,
    )

    print(
        "NO genera submit.",
        flush=True,
    )

    if not DATASET.exists():
        raise FileNotFoundError(
            DATASET
        )

    con = duckdb.connect()

    # ========================================================
    # ESQUEMA
    # ========================================================

    schema = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet(
            '{DATASET}'
        )
        """
    ).df()

    columns = (
        schema[
            "column_name"
        ]
        .astype(str)
        .tolist()
    )

    print(
        f"\nColumnas dataset: "
        f"{len(columns):,}",
        flush=True,
    )

    # ========================================================
    # IDENTIFICAR FEATURES
    # ========================================================

    metadata_cols = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
        LAG1_AVAILABLE_COL,
        LAG2_AVAILABLE_COL,
    }

    originales = [
        c
        for c in columns
        if (
            c not in metadata_cols
            and not c.endswith(
                "_lag1"
            )
            and not c.endswith(
                "_delta_lag1"
            )
            and not c.endswith(
                "_lag2"
            )
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

    if (
        len(originales)
        != EXPECTED_ORIGINALS
    ):
        raise ValueError(
            f"Originales esperadas: "
            f"{EXPECTED_ORIGINALS}; "
            f"encontradas: "
            f"{len(originales)}"
        )

    for feature_list, label in [
        (
            lag1,
            "lag1",
        ),
        (
            delta1,
            "delta_lag1",
        ),
        (
            lag2,
            "lag2",
        ),
    ]:

        missing = [
            c
            for c in feature_list
            if c not in columns
        ]

        if missing:
            raise ValueError(
                f"Faltan columnas {label}: "
                + ", ".join(
                    missing[:10]
                )
            )

    if (
        LAG1_AVAILABLE_COL
        not in columns
    ):
        raise ValueError(
            f"Falta "
            f"{LAG1_AVAILABLE_COL}"
        )

    if (
        LAG2_AVAILABLE_COL
        not in columns
    ):
        raise ValueError(
            f"Falta "
            f"{LAG2_AVAILABLE_COL}"
        )

    # FULL histórico actual de z527.
    full_features = (
        originales
        + lag1
        + delta1
        + [
            LAG1_AVAILABLE_COL
        ]
    )

    # Única diferencia experimental:
    # agregar las 152 variables lag2.
    full_lag2_features = (
        full_features
        + lag2
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
        len(full_lag2_features)
        != EXPECTED_FULL_LAG2
    ):
        raise ValueError(
            f"FULL+LAG2 debería tener "
            f"{EXPECTED_FULL_LAG2} features "
            f"y tiene "
            f"{len(full_lag2_features)}."
        )

    print(
        "\nFeatures:",
        flush=True,
    )

    print(
        f"  Originales       : "
        f"{len(originales)}",
        flush=True,
    )

    print(
        f"  Lag1             : "
        f"{len(lag1)}",
        flush=True,
    )

    print(
        f"  Delta lag1       : "
        f"{len(delta1)}",
        flush=True,
    )

    print(
        f"  FULL             : "
        f"{len(full_features)}",
        flush=True,
    )

    print(
        f"  Lag2 adicionales : "
        f"{len(lag2)}",
        flush=True,
    )

    print(
        f"  FULL + LAG2      : "
        f"{len(full_lag2_features)}",
        flush=True,
    )

    print(
        f"  {LAG2_AVAILABLE_COL} : "
        f"solo auditoría",
        flush=True,
    )

    # ========================================================
    # CARGAR EXCLUSIVAMENTE TRAIN / TEST
    # ========================================================

    required_features = (
        full_lag2_features
    )

    # IMPORTANTE:
    # lag2_disponible se carga para la auditoría de cobertura,
    # pero NO forma parte de full_features ni de
    # full_lag2_features.
    select_cols = [
        ID_COL,
        MONTH_COL,
        TARGET_COL,
        LAG2_AVAILABLE_COL,
    ] + required_features

    # Verificación defensiva de duplicados en el SELECT.
    duplicated_select_cols = [
        c
        for c in set(select_cols)
        if select_cols.count(c) > 1
    ]

    if duplicated_select_cols:
        raise ValueError(
            "Columnas duplicadas en SELECT: "
            + ", ".join(
                sorted(
                    duplicated_select_cols
                )
            )
        )

    select_sql = ",\n".join(
        q(c)
        for c in select_cols
    )

    months_sql = ", ".join(
        str(x)
        for x in (
            TRAIN_MONTHS
            + [TEST_MONTH]
        )
    )

    print(
        "\nCargando train/test...",
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

    # ========================================================
    # TARGET BINARIO
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

    # ========================================================
    # AUDITORÍA DE DISPONIBILIDAD LAG2
    # ========================================================

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "DISPONIBILIDAD LAG2",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    lag2_availability = (
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
        lag2_availability
        .iterrows()
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

    lag2_availability.to_csv(
        OUTPUT_DIR
        / "lag2_availability.csv",
        index=False,
    )

    # Validaciones específicas del diseño.
    coverage_by_month = {
        int(row[MONTH_COL]):
            int(row["sum"])
        for _, row
        in lag2_availability.iterrows()
    }

    if (
        coverage_by_month.get(
            202104,
            -1,
        )
        != 0
    ):
        raise ValueError(
            "202104 no debería "
            "tener lag2 disponible."
        )

    if (
        coverage_by_month.get(
            202105,
            0,
        )
        <= 0
    ):
        raise ValueError(
            "202105 debería "
            "tener lag2 disponible."
        )

    if (
        coverage_by_month.get(
            202106,
            0,
        )
        <= 0
    ):
        raise ValueError(
            "202106 debería "
            "tener lag2 disponible."
        )

    # ========================================================
    # A: FULL457
    # ========================================================

    summary_full, curve_full, imp_full = (
        evaluate_model(
            name="FULL457",
            feature_names=full_features,
            train_df=train_df,
            test_df=test_df,
        )
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
    # B: FULL + LAG2
    # ========================================================

    summary_lag2, curve_lag2, imp_lag2 = (
        evaluate_model(
            name="FULL_LAG2_609",
            feature_names=full_lag2_features,
            train_df=train_df,
            test_df=test_df,
        )
    )

    curve_lag2.to_csv(
        OUTPUT_DIR
        / "gain_curve_FULL_LAG2_609.csv",
        index=False,
    )

    imp_lag2.to_csv(
        OUTPUT_DIR
        / "importance_FULL_LAG2_609.csv",
        index=False,
    )

    # ========================================================
    # COMPARACIÓN
    # ========================================================

    summaries = pd.DataFrame(
        [
            summary_full,
            summary_lag2,
        ]
    )

    summaries.to_csv(
        OUTPUT_DIR
        / "summary_z533.csv",
        index=False,
    )

    base_gain = (
        summary_full[
            "gain_n12000_millions"
        ]
    )

    lag2_gain = (
        summary_lag2[
            "gain_n12000_millions"
        ]
    )

    delta_gain = (
        lag2_gain
        - base_gain
    )

    delta_pct = (
        100.0
        * delta_gain
        / base_gain
        if base_gain != 0
        else np.nan
    )

    delta_auc = (
        summary_lag2["auc"]
        - summary_full["auc"]
    )

    delta_ap = (
        summary_lag2[
            "average_precision"
        ]
        - summary_full[
            "average_precision"
        ]
    )

    delta_logloss = (
        summary_lag2[
            "log_loss"
        ]
        - summary_full[
            "log_loss"
        ]
    )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "COMPARACIÓN FINAL",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"FULL457 N12000       : "
        f"{base_gain:.3f} M",
        flush=True,
    )

    print(
        f"FULL+LAG2 N12000     : "
        f"{lag2_gain:.3f} M",
        flush=True,
    )

    print(
        f"Delta ganancia       : "
        f"{delta_gain:+.3f} M",
        flush=True,
    )

    print(
        f"Delta ganancia %     : "
        f"{delta_pct:+.2f}%",
        flush=True,
    )

    print(
        f"Delta AUC            : "
        f"{delta_auc:+.6f}",
        flush=True,
    )

    print(
        f"Delta AP             : "
        f"{delta_ap:+.6f}",
        flush=True,
    )

    print(
        f"Delta LogLoss        : "
        f"{delta_logloss:+.6f}",
        flush=True,
    )

    print(
        "\nMejor corte FULL     : "
        f"{summary_full['best_cut']:,} "
        f"| "
        f"{summary_full['best_gain_millions']:.3f} M",
        flush=True,
    )

    print(
        "Mejor corte FULL+LAG2: "
        f"{summary_lag2['best_cut']:,} "
        f"| "
        f"{summary_lag2['best_gain_millions']:.3f} M",
        flush=True,
    )

    # ========================================================
    # IMPORTANCIA ESPECÍFICA DE LAG2
    # ========================================================

    lag2_importance = (
        imp_lag2[
            imp_lag2["feature"]
            .str.endswith(
                "_lag2"
            )
        ]
        .copy()
    )

    total_gain_importance = (
        imp_lag2[
            "importance_gain"
        ].sum()
    )

    lag2_gain_importance = (
        lag2_importance[
            "importance_gain"
        ].sum()
    )

    lag2_importance_share = (
        lag2_gain_importance
        / total_gain_importance
        if total_gain_importance > 0
        else np.nan
    )

    print(
        f"\nParticipación lag2 en "
        f"gain importance: "
        f"{100 * lag2_importance_share:.2f}%",
        flush=True,
    )

    print(
        "\nTop 15 features lag2:",
        flush=True,
    )

    for _, row in (
        lag2_importance
        .head(15)
        .iterrows()
    ):

        share = (
            row["importance_gain"]
            / total_gain_importance
            if total_gain_importance > 0
            else 0
        )

        print(
            f"  "
            f"{row['feature']:<45} "
            f"{100 * share:8.4f}%",
            flush=True,
        )

    lag2_importance.to_csv(
        OUTPUT_DIR
        / "importance_lag2_only.csv",
        index=False,
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "script":
            "z533_competencia_01_lag2_screening.py",

        "dataset":
            str(DATASET),

        "train_months":
            TRAIN_MONTHS,

        "test_month":
            TEST_MONTH,

        "seed":
            SEED,

        "primary_cut":
            PRIMARY_CUT,

        "full_features":
            len(full_features),

        "full_lag2_features":
            len(full_lag2_features),

        "lag2_features_added":
            len(lag2),

        "lag1_available_used_as_feature":
            True,

        "lag2_available_loaded_for_audit":
            True,

        "lag2_available_used_as_feature":
            False,

        "uses_202108":
            False,

        "generates_submit":
            False,

        "params":
            PARAMS,

        "comparison": {
            "full_gain_n12000_millions":
                base_gain,

            "lag2_gain_n12000_millions":
                lag2_gain,

            "delta_gain_millions":
                delta_gain,

            "delta_gain_pct":
                delta_pct,

            "delta_auc":
                delta_auc,

            "delta_average_precision":
                delta_ap,

            "delta_log_loss":
                delta_logloss,

            "lag2_gain_importance_share":
                lag2_importance_share,
        },
    }

    with (
        OUTPUT_DIR
        / "metadata_z533.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL
    # ========================================================

    elapsed = (
        time.time()
        - start
    )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "FIN z533",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Tiempo total: "
        f"{elapsed / 60:.2f} min",
        flush=True,
    )

    print(
        f"\nResultados:\n  "
        f"{OUTPUT_DIR}",
        flush=True,
    )

    print(
        "\nNO se utilizó 202108.",
        flush=True,
    )

    print(
        "NO se generaron submits.",
        flush=True,
    )


if __name__ == "__main__":
    main()