#!/usr/bin/env python3

"""
z536_competencia_01_ponderacion_temporal.py

Screening controlado de ponderacion temporal.

Objetivo:
    Evaluar si dar menos peso a 202104 que a 202105 mejora
    la generalizacion hacia 202106.

Diseno:
    Train   : 202104 + 202105
    Test    : 202106
    Target  : BAJA+2
    Features: FULL457
    Seed    : 290497
    Orden   : foto_mes, numero_de_cliente

Pesos:
    202104 = 1.00, 0.75, 0.50, 0.25
    202105 = 1.00 siempre

No usa 202108.
No genera submits.
No cambia features ni hiperparametros.
"""

from pathlib import Path
import gc
import json
import time

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
# CONFIGURACION
# ============================================================

DATASET = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/feature_engineering_z523/"
    "competencia_01_historico_lag1.parquet"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/ponderacion_temporal_z536"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
LAG1_AVAILABLE_COL = "lag1_disponible"

TRAIN_MONTHS = [202104, 202105]
TEST_MONTH = 202106

OLD_MONTH = 202104
RECENT_MONTH = 202105

OLD_MONTH_WEIGHTS = [
    1.00,
    0.75,
    0.50,
    0.25,
]

SEED = 290497

PRIMARY_CUT = 12000
CUTS = list(range(8000, 16001, 500))

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

def quote_col(name):
    return '"' + name.replace('"', '""') + '"'


def evaluate_gain(y_true, probabilities, cut):
    order = np.argsort(
        -probabilities,
        kind="stable",
    )

    selected = order[:cut]

    positives = int(
        y_true[selected].sum()
    )

    negatives = int(
        cut - positives
    )

    gain = (
        positives * GAIN_POSITIVE
        + negatives * GAIN_NEGATIVE
    )

    return {
        "cut": int(cut),
        "positives": positives,
        "negatives": negatives,
        "gain": int(gain),
        "gain_millions": gain / 1_000_000,
    }


def train_model(
    weight_old,
    features,
    train,
    test,
):
    label = f"W{weight_old:.2f}"

    print()
    print("=" * 80)
    print(
        f"{label}: "
        f"{OLD_MONTH}={weight_old:.2f} | "
        f"{RECENT_MONTH}=1.00"
    )
    print("=" * 80)

    X_train = train[features]
    y_train = train["target_binario"].to_numpy(
        dtype=np.int8
    )

    X_test = test[features]
    y_test = test["target_binario"].to_numpy(
        dtype=np.int8
    )

    weights = np.where(
        train[MONTH_COL].to_numpy() == OLD_MONTH,
        weight_old,
        1.0,
    ).astype(np.float64)

    print(f"Features: {len(features)}")
    print(f"Train: {X_train.shape}")
    print(f"Test : {X_test.shape}")

    print(
        f"Peso efectivo total: "
        f"{weights.sum():,.2f}"
    )

    model = lgb.LGBMClassifier(
        **PARAMS
    )

    start = time.time()

    model.fit(
        X_train,
        y_train,
        sample_weight=weights,
    )

    elapsed = time.time() - start

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

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

    curve = pd.DataFrame(
        [
            evaluate_gain(
                y_test,
                probabilities,
                cut,
            )
            for cut in CUTS
        ]
    )

    primary = curve.loc[
        curve["cut"] == PRIMARY_CUT
    ].iloc[0]

    best = (
        curve
        .sort_values(
            ["gain", "cut"],
            ascending=[False, True],
        )
        .iloc[0]
    )

    print(f"AUC     : {auc:.6f}")
    print(f"AP      : {ap:.6f}")
    print(f"LogLoss : {ll:.6f}")
    print(f"Tiempo  : {elapsed:.1f}s")

    print(
        f"N={PRIMARY_CUT:,}: "
        f"positivos={int(primary['positives'])} "
        f"| gain="
        f"{primary['gain_millions']:.3f} M"
    )

    print(
        f"Mejor diagnostico: "
        f"N={int(best['cut']):,} "
        f"| positivos={int(best['positives'])} "
        f"| gain={best['gain_millions']:.3f} M"
    )

    summary = {
        "label": label,
        "weight_202104": float(weight_old),
        "weight_202105": 1.0,
        "auc": float(auc),
        "average_precision": float(ap),
        "log_loss": float(ll),
        "training_seconds": float(elapsed),
        "primary_cut": PRIMARY_CUT,
        "primary_positives":
            int(primary["positives"]),
        "primary_gain_millions":
            float(primary["gain_millions"]),
        "best_cut":
            int(best["cut"]),
        "best_positives":
            int(best["positives"]),
        "best_gain_millions":
            float(best["gain_millions"]),
    }

    del X_train
    del X_test
    del weights
    del model

    gc.collect()

    return summary, curve, probabilities


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "z536 - SCREENING DE PONDERACION TEMPORAL"
    )
    print("=" * 80)

    print(f"Dataset: {DATASET}")

    # --------------------------------------------------------
    # ESQUEMA
    # --------------------------------------------------------

    con = duckdb.connect()

    schema = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet('{DATASET}')
        """
    ).df()

    columns = schema[
        "column_name"
    ].tolist()

    metadata = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
        LAG1_AVAILABLE_COL,
    }

    originales = [
        c
        for c in columns
        if c not in metadata
        and not c.endswith("_lag1")
        and not c.endswith("_delta_lag1")
    ]

    lag1 = [
        f"{c}_lag1"
        for c in originales
    ]

    delta1 = [
        f"{c}_delta_lag1"
        for c in originales
    ]

    if len(originales) != EXPECTED_ORIGINALS:
        raise ValueError(
            f"Originales: esperaba "
            f"{EXPECTED_ORIGINALS}, "
            f"encontre {len(originales)}"
        )

    missing_lag1 = [
        c for c in lag1
        if c not in columns
    ]

    missing_delta = [
        c for c in delta1
        if c not in columns
    ]

    if missing_lag1:
        raise ValueError(
            f"Faltan lag1: "
            f"{missing_lag1[:10]}"
        )

    if missing_delta:
        raise ValueError(
            f"Faltan delta1: "
            f"{missing_delta[:10]}"
        )

    features = (
        originales
        + lag1
        + delta1
        + [LAG1_AVAILABLE_COL]
    )

    if len(features) != EXPECTED_FULL:
        raise ValueError(
            f"FULL: esperaba {EXPECTED_FULL}, "
            f"encontre {len(features)}"
        )

    print()
    print(f"Originales : {len(originales)}")
    print(f"Lag1       : {len(lag1)}")
    print(f"Delta lag1 : {len(delta1)}")
    print(f"FULL       : {len(features)}")

    # --------------------------------------------------------
    # CARGA
    # --------------------------------------------------------

    required = [
        ID_COL,
        MONTH_COL,
        TARGET_COL,
    ] + features

    select_sql = ",\n".join(
        quote_col(c)
        for c in required
    )

    print()
    print(
        "Cargando datos en orden canonico..."
    )

    start = time.time()

    df = con.execute(
        f"""
        SELECT
            {select_sql}
        FROM read_parquet('{DATASET}')
        WHERE {quote_col(MONTH_COL)}
              IN (202104, 202105, 202106)
        ORDER BY
            {quote_col(MONTH_COL)},
            {quote_col(ID_COL)}
        """
    ).df()

    con.close()

    print(
        f"Carga completada en "
        f"{time.time() - start:.1f}s"
    )

    print(f"Filas: {len(df):,}")

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    df["target_binario"] = (
        df[TARGET_COL] == "BAJA+2"
    ).astype(np.int8)

    train = (
        df[
            df[MONTH_COL].isin(
                TRAIN_MONTHS
            )
        ]
        .reset_index(drop=True)
    )

    test = (
        df[
            df[MONTH_COL] == TEST_MONTH
        ]
        .reset_index(drop=True)
    )

    print()
    print(f"Train rows: {len(train):,}")
    print(f"Test rows : {len(test):,}")

    print(
        f"Train BAJA+2: "
        f"{int(train['target_binario'].sum()):,}"
    )

    print(
        f"Test BAJA+2 : "
        f"{int(test['target_binario'].sum()):,}"
    )

    audit_months = (
        train
        .groupby(MONTH_COL)["target_binario"]
        .agg(["count", "sum"])
        .reset_index()
    )

    print()
    print("Train por mes:")
    print(
        audit_months.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # MODELOS
    # --------------------------------------------------------

    summaries = []
    curves = {}
    probabilities = {}

    for weight in OLD_MONTH_WEIGHTS:

        summary, curve, probs = train_model(
            weight_old=weight,
            features=features,
            train=train,
            test=test,
        )

        label = summary["label"]

        summaries.append(summary)
        curves[label] = curve
        probabilities[label] = probs

        curve.to_csv(
            OUTPUT_DIR
            / f"gain_curve_{label}.csv",
            index=False,
        )

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summaries
    )

    control_gain = float(
        summary_df.loc[
            summary_df["weight_202104"] == 1.0,
            "primary_gain_millions",
        ].iloc[0]
    )

    summary_df[
        "delta_vs_control_millions"
    ] = (
        summary_df["primary_gain_millions"]
        - control_gain
    )

    summary_df[
        "delta_vs_control_pct"
    ] = (
        100.0
        * summary_df[
            "delta_vs_control_millions"
        ]
        / control_gain
    )

    print()
    print("=" * 80)
    print("RESUMEN N=12.000")
    print("=" * 80)

    print(
        summary_df[
            [
                "weight_202104",
                "weight_202105",
                "auc",
                "average_precision",
                "log_loss",
                "primary_positives",
                "primary_gain_millions",
                "delta_vs_control_millions",
                "delta_vs_control_pct",
                "best_cut",
                "best_gain_millions",
            ]
        ].to_string(index=False)
    )

    summary_df.to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # CURVAS COMPARADAS
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        {"cut": CUTS}
    )

    for weight in OLD_MONTH_WEIGHTS:

        label = f"W{weight:.2f}"
        curve = curves[label]

        comparison[
            f"positives_{label}"
        ] = curve[
            "positives"
        ].to_numpy()

        comparison[
            f"gain_{label}_millions"
        ] = curve[
            "gain_millions"
        ].to_numpy()

    for weight in OLD_MONTH_WEIGHTS[1:]:

        label = f"W{weight:.2f}"

        comparison[
            f"delta_{label}_vs_W1.00"
        ] = (
            comparison[
                f"gain_{label}_millions"
            ]
            - comparison[
                "gain_W1.00_millions"
            ]
        )

    print()
    print("=" * 80)
    print("COMPARACION POR CUT")
    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "comparison_by_cut.csv",
        index=False,
    )

    # --------------------------------------------------------
    # DIVERSIDAD DE RANKING VS CONTROL
    # --------------------------------------------------------

    control_probs = probabilities[
        "W1.00"
    ]

    control_order = np.argsort(
        -control_probs,
        kind="stable",
    )

    control_ids = set(
        test.iloc[
            control_order[:PRIMARY_CUT]
        ][ID_COL]
        .astype(int)
        .tolist()
    )

    diversity = []

    print()
    print("=" * 80)
    print("DIVERSIDAD VS W1.00")
    print("=" * 80)

    for weight in OLD_MONTH_WEIGHTS[1:]:

        label = f"W{weight:.2f}"
        probs = probabilities[label]

        corr = float(
            np.corrcoef(
                control_probs,
                probs,
            )[0, 1]
        )

        max_diff = float(
            np.max(
                np.abs(
                    control_probs - probs
                )
            )
        )

        order = np.argsort(
            -probs,
            kind="stable",
        )

        ids = set(
            test.iloc[
                order[:PRIMARY_CUT]
            ][ID_COL]
            .astype(int)
            .tolist()
        )

        jaccard = (
            len(control_ids & ids)
            / len(control_ids | ids)
        )

        diversity.append(
            {
                "label": label,
                "weight_202104":
                    float(weight),
                "correlation":
                    corr,
                "max_abs_diff":
                    max_diff,
                "jaccard_top12000":
                    jaccard,
            }
        )

        print(
            f"{label}: "
            f"corr={corr:.9f} "
            f"| max_diff={max_diff:.9f} "
            f"| Jaccard={jaccard:.6f}"
        )

    pd.DataFrame(
        diversity
    ).to_csv(
        OUTPUT_DIR
        / "diversity_vs_control.csv",
        index=False,
    )

    # --------------------------------------------------------
    # AUDITORIA
    # --------------------------------------------------------

    audit = {
        "experiment": "z536",
        "dataset": str(DATASET),
        "train_months": TRAIN_MONTHS,
        "test_month": TEST_MONTH,
        "target": "BAJA+2",
        "features": len(features),
        "ordering": [
            MONTH_COL,
            ID_COL,
        ],
        "seed": SEED,
        "old_month": OLD_MONTH,
        "recent_month": RECENT_MONTH,
        "old_month_weights":
            OLD_MONTH_WEIGHTS,
        "recent_month_weight": 1.0,
        "primary_cut": PRIMARY_CUT,
        "cuts": CUTS,
        "params": PARAMS,
    }

    with open(
        OUTPUT_DIR / "audit.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # GANADOR
    # --------------------------------------------------------

    winner = (
        summary_df
        .sort_values(
            [
                "primary_gain_millions",
                "weight_202104",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .iloc[0]
    )

    print()
    print("=" * 80)
    print("RESULTADO PRINCIPAL")
    print("=" * 80)

    print(
        f"Control W1.00: "
        f"{control_gain:.3f} M"
    )

    print(
        f"Mejor peso 202104: "
        f"{winner['weight_202104']:.2f}"
    )

    print(
        f"Gain N={PRIMARY_CUT:,}: "
        f"{winner['primary_gain_millions']:.3f} M"
    )

    print(
        f"Delta vs control: "
        f"{winner['delta_vs_control_millions']:+.3f} M "
        f"({winner['delta_vs_control_pct']:+.2f}%)"
    )

    print()
    print(
        f"Archivos: {OUTPUT_DIR}"
    )

    print()
    print("z536 finalizado.")


if __name__ == "__main__":
    main()