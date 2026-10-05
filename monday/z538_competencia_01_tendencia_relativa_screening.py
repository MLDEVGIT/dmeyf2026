#!/usr/bin/env python3

"""
z538_competencia_01_tendencia_relativa_screening.py

Screening controlado de tendencias relativas sobre variables importantes.

Hipotesis:
    FULL457 ya contiene:
      - x_t
      - x_t-1
      - delta absoluto = x_t - x_t-1

    Agregamos, solamente para variables base importantes:

      delta_rel =
          (x_t - x_t-1)
          / (abs(x_t) + abs(x_t-1) + 1)

    Esto representa direccion e intensidad relativa del cambio,
    evitando explosiones cuando lag1 esta cerca de cero.

Seleccion:
    Top 20 variables base del ranking z530.
    Se excluyen automaticamente variables temporales/edad.

Experimento:
    A: FULL457
    B: FULL457 + DELTA_REL

Train:
    202104 + 202105

Test:
    202106

Seed:
    290497

Orden:
    foto_mes, numero_de_cliente

No usa 202108.
No genera submits.
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

RANKING_BASE = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/importancia_historicas_z530/"
    "ranking_variables_base.csv"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/tendencia_relativa_z538"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
LAG_AVAILABLE_COL = "lag1_disponible"

TRAIN_MONTHS = [
    202104,
    202105,
]

TEST_MONTH = 202106

SEED = 290497

TOP_BASE = 20

# Exclusiones conceptuales:
# no tiene sentido interpretar edad/fechas como
# variaciones relativas financieras u operativas.
EXCLUDE_RELATIVE = {
    "cliente_edad",
    "cliente_antiguedad",
}

EXCLUDE_NAME_PATTERNS = (
    "fecha",
    "fvencimiento",
)

PRIMARY_CUT = 12000

CUTS = list(
    range(
        8000,
        16001,
        500,
    )
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
    "importance_type": "gain",
}


# ============================================================
# HELPERS
# ============================================================

def quote_col(name):
    return '"' + name.replace('"', '""') + '"'


def gain_at_cut(
    y_true,
    probabilities,
    cut,
):
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
        "gain_millions":
            gain / 1_000_000,
    }


def evaluate_model(
    name,
    features,
    train,
    test,
):

    print()
    print("=" * 80)
    print(f"MODELO: {name}")
    print("=" * 80)

    X_train = train[features]
    X_test = test[features]

    y_train = (
        train["target_binario"]
        .to_numpy(dtype=np.int8)
    )

    y_test = (
        test["target_binario"]
        .to_numpy(dtype=np.int8)
    )

    print(f"Features: {len(features)}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")
    print(
        f"Train positivos: "
        f"{int(y_train.sum()):,}"
    )
    print(
        f"Test positivos : "
        f"{int(y_test.sum()):,}"
    )

    model = lgb.LGBMClassifier(
        **PARAMS
    )

    start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    elapsed = time.time() - start

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
            gain_at_cut(
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
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    importance = pd.DataFrame(
        {
            "feature": features,
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
    ).sort_values(
        "importance_gain",
        ascending=False,
    )

    print(f"AUC      : {auc:.6f}")
    print(f"AP       : {ap:.6f}")
    print(f"LogLoss  : {ll:.6f}")
    print(f"Tiempo   : {elapsed:.1f}s")

    print(
        f"N={PRIMARY_CUT:,}: "
        f"positivos="
        f"{int(primary['positives'])} "
        f"| gain="
        f"{primary['gain_millions']:.3f} M"
    )

    print(
        f"Mejor diagnostico: "
        f"N={int(best['cut']):,} "
        f"| positivos="
        f"{int(best['positives'])} "
        f"| gain="
        f"{best['gain_millions']:.3f} M"
    )

    summary = {
        "model": name,
        "features":
            len(features),
        "auc":
            float(auc),
        "average_precision":
            float(ap),
        "log_loss":
            float(ll),
        "training_seconds":
            float(elapsed),
        "primary_cut":
            PRIMARY_CUT,
        "primary_positives":
            int(primary["positives"]),
        "primary_gain_millions":
            float(
                primary[
                    "gain_millions"
                ]
            ),
        "best_cut":
            int(best["cut"]),
        "best_positives":
            int(best["positives"]),
        "best_gain_millions":
            float(
                best[
                    "gain_millions"
                ]
            ),
    }

    del X_train
    del X_test
    del model

    gc.collect()

    return (
        summary,
        curve,
        probabilities,
        importance,
    )


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
        "z538 - SCREENING DE TENDENCIA RELATIVA"
    )
    print("=" * 80)

    print(f"Dataset: {DATASET}")
    print(f"Ranking : {RANKING_BASE}")

    # ========================================================
    # ESQUEMA
    # ========================================================

    con = duckdb.connect()

    schema = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet('{DATASET}')
        """
    ).df()

    columns = (
        schema["column_name"]
        .tolist()
    )

    metadata = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
        LAG_AVAILABLE_COL,
    }

    originales = [
        c
        for c in columns
        if c not in metadata
        and not c.endswith(
            "_lag1"
        )
        and not c.endswith(
            "_delta_lag1"
        )
    ]

    if (
        len(originales)
        != EXPECTED_ORIGINALS
    ):
        raise ValueError(
            "Cantidad inesperada "
            f"de originales: "
            f"{len(originales)}"
        )

    lag1 = [
        f"{c}_lag1"
        for c in originales
    ]

    delta1 = [
        f"{c}_delta_lag1"
        for c in originales
    ]

    missing_lag = [
        c
        for c in lag1
        if c not in columns
    ]

    missing_delta = [
        c
        for c in delta1
        if c not in columns
    ]

    if missing_lag:
        raise ValueError(
            f"Faltan lag1: "
            f"{missing_lag[:10]}"
        )

    if missing_delta:
        raise ValueError(
            f"Faltan delta1: "
            f"{missing_delta[:10]}"
        )

    features_full = (
        originales
        + lag1
        + delta1
        + [LAG_AVAILABLE_COL]
    )

    if (
        len(features_full)
        != EXPECTED_FULL
    ):
        raise ValueError(
            "FULL deberia tener "
            f"{EXPECTED_FULL} features "
            f"y tiene "
            f"{len(features_full)}."
        )

    print()
    print(
        f"Originales : "
        f"{len(originales)}"
    )
    print(
        f"Lag1       : "
        f"{len(lag1)}"
    )
    print(
        f"Delta lag1 : "
        f"{len(delta1)}"
    )
    print(
        f"FULL       : "
        f"{len(features_full)}"
    )

    # ========================================================
    # SELECCION TOP BASE z530
    # ========================================================

    ranking = pd.read_csv(
        RANKING_BASE
    ).sort_values(
        "ranking"
    )

    top = (
        ranking
        .head(TOP_BASE)
        .copy()
    )

    def eligible(variable):

        lower = (
            variable.lower()
        )

        if (
            variable
            in EXCLUDE_RELATIVE
        ):
            return False

        if any(
            pattern in lower
            for pattern
            in EXCLUDE_NAME_PATTERNS
        ):
            return False

        return True

    selected_base = [
        variable
        for variable
        in top["variable_base"]
        .tolist()
        if eligible(variable)
    ]

    excluded_base = [
        variable
        for variable
        in top["variable_base"]
        .tolist()
        if not eligible(variable)
    ]

    print()
    print("=" * 80)
    print("SELECCION DE VARIABLES")
    print("=" * 80)

    print(
        f"Top base considerado: "
        f"{TOP_BASE}"
    )

    print(
        f"Seleccionadas: "
        f"{len(selected_base)}"
    )

    print(
        f"Excluidas: "
        f"{len(excluded_base)}"
    )

    print()
    print("Seleccionadas:")

    for i, variable in enumerate(
        selected_base,
        start=1,
    ):
        rank = int(
            ranking.loc[
                ranking[
                    "variable_base"
                ] == variable,
                "ranking",
            ].iloc[0]
        )

        gain_pct = float(
            ranking.loc[
                ranking[
                    "variable_base"
                ] == variable,
                "gain_pct_mean",
            ].iloc[0]
        )

        print(
            f"  {i:02d}. "
            f"{variable:<32} "
            f"| rank={rank:2d} "
            f"| gain={gain_pct:.3f}%"
        )

    print()
    print(
        "Excluidas:",
        excluded_base,
    )

    # ========================================================
    # CARGA
    # ========================================================

    required = [
        ID_COL,
        MONTH_COL,
        TARGET_COL,
    ] + features_full

    select_sql = ",\n".join(
        quote_col(c)
        for c in required
    )

    print()
    print(
        "Cargando train/test "
        "en orden canonico..."
    )

    start = time.time()

    df = con.execute(
        f"""
        SELECT
            {select_sql}
        FROM read_parquet(
            '{DATASET}'
        )
        WHERE
            {quote_col(MONTH_COL)}
            IN (
                202104,
                202105,
                202106
            )
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

    print(
        f"Filas: {len(df):,}"
    )

    # ========================================================
    # TARGET
    # ========================================================

    df[
        "target_binario"
    ] = (
        df[TARGET_COL]
        == "BAJA+2"
    ).astype(np.int8)

    # ========================================================
    # TENDENCIAS RELATIVAS
    # ========================================================

    relative_features = []

    print()
    print(
        "Calculando tendencias "
        "relativas..."
    )

    for variable in selected_base:

        lag_col = (
            f"{variable}_lag1"
        )

        new_col = (
            f"{variable}_delta_rel"
        )

        current = (
            df[variable]
            .astype(np.float64)
        )

        previous = (
            df[lag_col]
            .astype(np.float64)
        )

        denominator = (
            np.abs(current)
            + np.abs(previous)
            + 1.0
        )

        df[new_col] = (
            (current - previous)
            / denominator
        )

        # Si no existe lag1,
        # la tendencia relativa tampoco existe.
        df.loc[
            previous.isna(),
            new_col,
        ] = np.nan

        relative_features.append(
            new_col
        )

    print(
        f"Nuevas features: "
        f"{len(relative_features)}"
    )

    features_relative = (
        features_full
        + relative_features
    )

    print(
        f"FULL + REL: "
        f"{len(features_relative)}"
    )

    # ========================================================
    # AUDITORIA DE NUEVAS FEATURES
    # ========================================================

    finite_stats = []

    for feature in relative_features:

        values = (
            df[feature]
            .to_numpy(
                dtype=np.float64
            )
        )

        finite = np.isfinite(
            values
        )

        finite_values = (
            values[finite]
        )

        if len(finite_values):

            min_value = float(
                np.min(
                    finite_values
                )
            )

            max_value = float(
                np.max(
                    finite_values
                )
            )

        else:

            min_value = np.nan
            max_value = np.nan

        finite_stats.append(
            {
                "feature": feature,
                "finite_rows":
                    int(
                        finite.sum()
                    ),
                "finite_pct":
                    float(
                        100
                        * finite.mean()
                    ),
                "min":
                    min_value,
                "max":
                    max_value,
            }
        )

    finite_df = pd.DataFrame(
        finite_stats
    )

    print()
    print(
        "Rango de tendencias relativas:"
    )

    print(
        finite_df[
            [
                "feature",
                "finite_pct",
                "min",
                "max",
            ]
        ].to_string(
            index=False
        )
    )

    if (
        finite_df["min"].min()
        < -1.0000001
        or finite_df["max"].max()
        > 1.0000001
    ):
        raise ValueError(
            "Hay delta_rel fuera "
            "del rango esperado [-1, 1]."
        )

    finite_df.to_csv(
        OUTPUT_DIR
        / "audit_relative_features.csv",
        index=False,
    )

    # ========================================================
    # TRAIN / TEST
    # ========================================================

    train = (
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

    test = (
        df[
            df[MONTH_COL]
            == TEST_MONTH
        ]
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        f"Train rows: "
        f"{len(train):,}"
    )

    print(
        f"Test rows : "
        f"{len(test):,}"
    )

    print(
        f"Train BAJA+2: "
        f"{int(train['target_binario'].sum()):,}"
    )

    print(
        f"Test BAJA+2 : "
        f"{int(test['target_binario'].sum()):,}"
    )

    # ========================================================
    # MODELO A - FULL457
    # ========================================================

    (
        summary_full,
        curve_full,
        probs_full,
        importance_full,
    ) = evaluate_model(
        name="FULL457",
        features=features_full,
        train=train,
        test=test,
    )

    # ========================================================
    # MODELO B - FULL + DELTA RELATIVO
    # ========================================================

    (
        summary_rel,
        curve_rel,
        probs_rel,
        importance_rel,
    ) = evaluate_model(
        name=(
            f"FULL_REL"
            f"{len(features_relative)}"
        ),
        features=features_relative,
        train=train,
        test=test,
    )

    # ========================================================
    # COMPARACION
    # ========================================================

    comparison = pd.DataFrame(
        {
            "cut":
                curve_full["cut"],
            "positives_FULL":
                curve_full[
                    "positives"
                ],
            "gain_FULL_millions":
                curve_full[
                    "gain_millions"
                ],
            "positives_REL":
                curve_rel[
                    "positives"
                ],
            "gain_REL_millions":
                curve_rel[
                    "gain_millions"
                ],
        }
    )

    comparison[
        "delta_REL_vs_FULL_millions"
    ] = (
        comparison[
            "gain_REL_millions"
        ]
        - comparison[
            "gain_FULL_millions"
        ]
    )

    print()
    print("=" * 80)
    print("COMPARACION")
    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )

    # ========================================================
    # DIVERSIDAD
    # ========================================================

    corr = float(
        np.corrcoef(
            probs_full,
            probs_rel,
        )[0, 1]
    )

    max_diff = float(
        np.max(
            np.abs(
                probs_full
                - probs_rel
            )
        )
    )

    order_full = np.argsort(
        -probs_full,
        kind="stable",
    )

    order_rel = np.argsort(
        -probs_rel,
        kind="stable",
    )

    ids_full = set(
        test.iloc[
            order_full[
                :PRIMARY_CUT
            ]
        ][ID_COL]
        .astype(int)
        .tolist()
    )

    ids_rel = set(
        test.iloc[
            order_rel[
                :PRIMARY_CUT
            ]
        ][ID_COL]
        .astype(int)
        .tolist()
    )

    jaccard = (
        len(
            ids_full
            & ids_rel
        )
        / len(
            ids_full
            | ids_rel
        )
    )

    print()
    print(
        "Diversidad de ranking:"
    )

    print(
        f"Correlacion probabilidades: "
        f"{corr:.9f}"
    )

    print(
        f"Maxima diferencia absoluta: "
        f"{max_diff:.9f}"
    )

    print(
        f"Jaccard top "
        f"{PRIMARY_CUT:,}: "
        f"{jaccard:.6f}"
    )

    # ========================================================
    # IMPORTANCIA NUEVAS VARIABLES
    # ========================================================

    relative_importance = (
        importance_rel[
            importance_rel[
                "feature"
            ].isin(
                relative_features
            )
        ]
        .copy()
    )

    total_gain = float(
        importance_rel[
            "importance_gain"
        ].sum()
    )

    relative_gain = float(
        relative_importance[
            "importance_gain"
        ].sum()
    )

    if total_gain > 0:

        relative_share = (
            100.0
            * relative_gain
            / total_gain
        )

    else:

        relative_share = 0.0

    print()
    print(
        "Share gain importance "
        f"delta_rel: "
        f"{relative_share:.2f}%"
    )

    print()
    print(
        "Importancia de "
        "tendencias relativas:"
    )

    print(
        relative_importance[
            [
                "feature",
                "importance_gain",
                "importance_split",
            ]
        ]
        .sort_values(
            "importance_gain",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # RESULTADO PRINCIPAL
    # ========================================================

    gain_full = (
        summary_full[
            "primary_gain_millions"
        ]
    )

    gain_rel = (
        summary_rel[
            "primary_gain_millions"
        ]
    )

    delta_gain = (
        gain_rel
        - gain_full
    )

    delta_pct = (
        100.0
        * delta_gain
        / gain_full
    )

    print()
    print("=" * 80)
    print("RESULTADO PRINCIPAL")
    print("=" * 80)

    print(
        f"FULL457 "
        f"N={PRIMARY_CUT:,}: "
        f"{gain_full:.3f} M"
    )

    print(
        f"FULL+REL "
        f"N={PRIMARY_CUT:,}: "
        f"{gain_rel:.3f} M"
    )

    print(
        f"Delta: "
        f"{delta_gain:+.3f} M "
        f"({delta_pct:+.2f}%)"
    )

    # ========================================================
    # GUARDADO
    # ========================================================

    summary_df = pd.DataFrame(
        [
            summary_full,
            summary_rel,
        ]
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    curve_full.to_csv(
        OUTPUT_DIR
        / "gain_curve_FULL457.csv",
        index=False,
    )

    curve_rel.to_csv(
        OUTPUT_DIR
        / "gain_curve_FULL_REL.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "comparison_by_cut.csv",
        index=False,
    )

    importance_full.to_csv(
        OUTPUT_DIR
        / "importance_FULL457.csv",
        index=False,
    )

    importance_rel.to_csv(
        OUTPUT_DIR
        / "importance_FULL_REL.csv",
        index=False,
    )

    relative_importance.to_csv(
        OUTPUT_DIR
        / "importance_relative_only.csv",
        index=False,
    )

    selected_df = (
        ranking[
            ranking[
                "variable_base"
            ].isin(
                selected_base
            )
        ]
        .copy()
        .sort_values(
            "ranking"
        )
    )

    selected_df.to_csv(
        OUTPUT_DIR
        / "selected_base_variables.csv",
        index=False,
    )

    metadata = {
        "script":
            "z538_competencia_01_"
            "tendencia_relativa_screening.py",
        "dataset":
            str(DATASET),
        "ranking_base":
            str(RANKING_BASE),
        "train_months":
            TRAIN_MONTHS,
        "test_month":
            TEST_MONTH,
        "target":
            "BAJA+2",
        "seed":
            SEED,
        "ordering": [
            MONTH_COL,
            ID_COL,
        ],
        "top_base":
            TOP_BASE,
        "selected_base":
            selected_base,
        "excluded_base":
            excluded_base,
        "relative_formula":
            "(x-lag1)/(abs(x)+abs(lag1)+1)",
        "full_features":
            len(features_full),
        "relative_features":
            len(relative_features),
        "total_features":
            len(features_relative),
        "primary_cut":
            PRIMARY_CUT,
        "cuts":
            CUTS,
        "params":
            PARAMS,
        "uses_202108":
            False,
        "generates_submit":
            False,
    }

    with open(
        OUTPUT_DIR
        / "metadata_z538.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        f"Archivos guardados en: "
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        "z538 finalizado."
    )


if __name__ == "__main__":
    main()