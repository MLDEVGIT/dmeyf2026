"""
z541_competencia_01_actividad_ablation.py

Hipotesis
---------
Construir primero una feature conductual agregada dentro de cada mes:

    actividad_canales_t =
        cantidad de tipos de transacciones con valor > 0

y luego generar historia sobre esa feature:

    actividad_t
    actividad_lag1
    actividad_lag2
    delta1
    delta2
    tendencia2
    promedio3
    min3
    max3
    rango3

Objetivo
--------
Comparar de forma controlada:

    BASE = FULL457 de z523
    ACT  = FULL457 + 10 features de actividad

Los lag2 se toman del dataset z532, que respeta exactamente
dos meses calendario.

No se utilizan targets para construir features.
No se utiliza 202108 para entrenamiento ni validacion.
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
    "competencia_01/feature_engineering_z532/"
    "competencia_01_historico_lag1_lag2.parquet"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/actividad_ablation_z541"
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
    "importance_type": "gain",
}


TRANSACTION_VARS = [
    "ctarjeta_debito_transacciones",
    "ctarjeta_visa_transacciones",
    "ctarjeta_master_transacciones",
    "cpayroll_trx",
    "cpayroll2_trx",
    "ccuenta_debitos_automaticos",
    "ctarjeta_visa_debitos_automaticos",
    "ctarjeta_master_debitos_automaticos",
    "cpagodeservicios",
    "cpagomiscuentas",
    "ccajeros_propios_descuentos",
    "ctarjeta_visa_descuentos",
    "ctarjeta_master_descuentos",
    "ccomisiones_mantenimiento",
    "ccomisiones_otras",
    "cforex_buy",
    "cforex_sell",
    "ctransferencias_recibidas",
    "ctransferencias_emitidas",
    "cextraccion_autoservicio",
    "ccheques_depositados",
    "ccheques_emitidos",
    "ccheques_depositados_rechazados",
    "ccheques_emitidos_rechazados",
    "ccallcenter_transacciones",
    "chomebanking_transacciones",
    "ccajas_transacciones",
    "ccajas_consultas",
    "ccajas_depositos",
    "ccajas_extracciones",
    "ccajas_otras",
    "catm_trx",
    "catm_trx_other",
    "cmobile_app_trx",
]


ACTIVITY_FEATURES = [
    "actividad_canales",
    "actividad_canales_lag1",
    "actividad_canales_lag2",
    "actividad_delta1",
    "actividad_delta2",
    "actividad_tendencia2",
    "actividad_promedio3",
    "actividad_min3",
    "actividad_max3",
    "actividad_rango3",
]


# ============================================================
# HELPERS
# ============================================================

def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


def gain_at_cut(
    y_true,
    probabilities,
    cut,
):
    order = np.argsort(
        -probabilities
    )

    selected = order[:cut]

    positives = int(
        y_true[selected].sum()
    )

    negatives = (
        len(selected)
        - positives
    )

    gain = (
        positives * GAIN_POSITIVE
        + negatives * GAIN_NEGATIVE
    )

    return {
        "cut": int(cut),
        "positives": positives,
        "negatives": int(negatives),
        "gain": float(gain),
        "gain_millions":
            float(gain / 1_000_000),
    }


def evaluate_model(
    name,
    train_df,
    test_df,
    features,
):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Features        : "
        f"{len(features):,}"
    )

    X_train = train_df[
        features
    ]

    X_test = test_df[
        features
    ]

    y_train = (
        train_df[TARGET_COL]
        .eq("BAJA+2")
        .astype(np.int8)
        .to_numpy()
    )

    y_test = (
        test_df[TARGET_COL]
        .eq("BAJA+2")
        .astype(np.int8)
        .to_numpy()
    )

    print(
        f"Train filas    : "
        f"{len(X_train):,}"
    )

    print(
        f"Test filas     : "
        f"{len(X_test):,}"
    )

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

    elapsed = (
        time.time()
        - start
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
        curve["cut"]
        == PRIMARY_CUT
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
            "feature":
                features,
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

    print(
        f"AUC      : {auc:.6f}"
    )

    print(
        f"AP       : {ap:.6f}"
    )

    print(
        f"LogLoss  : {ll:.6f}"
    )

    print(
        f"Tiempo   : {elapsed:.1f}s"
    )

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
        "z541 - ABLATION ACTIVIDAD HISTORICA"
    )
    print("=" * 80)

    print(
        f"Dataset: {DATASET}"
    )

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
        LAG1_AVAILABLE_COL,
        LAG2_AVAILABLE_COL,
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
        and not c.endswith(
            "_lag2"
        )
    ]

    if (
        len(originales)
        != EXPECTED_ORIGINALS
    ):
        raise ValueError(
            "Cantidad inesperada "
            "de originales: "
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

    features_full = (
        originales
        + lag1
        + delta1
        + [LAG1_AVAILABLE_COL]
    )

    if (
        len(features_full)
        != EXPECTED_FULL
    ):
        raise ValueError(
            f"FULL deberia tener "
            f"{EXPECTED_FULL} features "
            f"y tiene "
            f"{len(features_full)}."
        )

    # --------------------------------------------------------
    # Validacion variables transaccionales
    # --------------------------------------------------------

    missing = []

    for c in TRANSACTION_VARS:

        for suffix in [
            "",
            "_lag1",
            "_lag2",
        ]:
            candidate = (
                c + suffix
            )

            if candidate not in columns:
                missing.append(
                    candidate
                )

    if missing:
        raise ValueError(
            "Faltan columnas "
            "transaccionales: "
            f"{missing}"
        )

    print()
    print(
        f"Originales          : "
        f"{len(originales)}"
    )
    print(
        f"FULL baseline       : "
        f"{len(features_full)}"
    )
    print(
        f"Variables actividad : "
        f"{len(TRANSACTION_VARS)}"
    )

    # ========================================================
    # CONSTRUCCION DE ACTIVIDAD
    # ========================================================

    def activity_expression(
        suffix="",
    ):
        return " + ".join(
            [
                (
                    "CASE WHEN "
                    f"{quote_identifier(c + suffix)} "
                    "> 0 "
                    "THEN 1 ELSE 0 END"
                )
                for c
                in TRANSACTION_VARS
            ]
        )

    act0 = activity_expression("")
    act1 = activity_expression(
        "_lag1"
    )
    act2 = activity_expression(
        "_lag2"
    )

    query = f"""
        WITH base AS (
            SELECT
                *,
                ({act0})
                    AS actividad_canales,

                CASE
                    WHEN {quote_identifier(LAG1_AVAILABLE_COL)} = 1
                    THEN ({act1})
                    ELSE NULL
                END
                    AS actividad_canales_lag1,

                CASE
                    WHEN {quote_identifier(LAG2_AVAILABLE_COL)} = 1
                    THEN ({act2})
                    ELSE NULL
                END
                    AS actividad_canales_lag2

            FROM read_parquet(
                '{DATASET}'
            )
        ),

        engineered AS (
            SELECT
                *,

                actividad_canales
                    - actividad_canales_lag1
                    AS actividad_delta1,

                actividad_canales_lag1
                    - actividad_canales_lag2
                    AS actividad_delta2,

                actividad_canales
                    - actividad_canales_lag2
                    AS actividad_tendencia2,

                (
                    actividad_canales
                    + actividad_canales_lag1
                    + actividad_canales_lag2
                ) / 3.0
                    AS actividad_promedio3,

                LEAST(
                    actividad_canales,
                    actividad_canales_lag1,
                    actividad_canales_lag2
                )
                    AS actividad_min3,

                GREATEST(
                    actividad_canales,
                    actividad_canales_lag1,
                    actividad_canales_lag2
                )
                    AS actividad_max3

            FROM base
        )

        SELECT
            *,
            actividad_max3
                - actividad_min3
                AS actividad_rango3

        FROM engineered

        WHERE {quote_identifier(MONTH_COL)}
            IN (
                202104,
                202105,
                202106
            )

        ORDER BY
            {quote_identifier(MONTH_COL)},
            {quote_identifier(ID_COL)}
    """

    print()
    print(
        "Construyendo features "
        "de actividad..."
    )

    start = time.time()

    df = con.execute(
        query
    ).df()

    print(
        f"Filas cargadas: "
        f"{len(df):,}"
    )

    print(
        f"Tiempo       : "
        f"{time.time() - start:.1f}s"
    )

    # ========================================================
    # AUDITORIA
    # ========================================================

    print()
    print("=" * 80)
    print(
        "AUDITORIA ACTIVIDAD"
    )
    print("=" * 80)

    audit = (
        df.groupby(
            [
                MONTH_COL,
                TARGET_COL,
            ],
            dropna=False,
        )[
            [
                "actividad_canales",
                "actividad_canales_lag1",
                "actividad_canales_lag2",
                "actividad_delta1",
                "actividad_delta2",
                "actividad_tendencia2",
            ]
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
            ]
        )
    )

    print(
        audit.to_string()
    )

    audit.to_csv(
        OUTPUT_DIR
        / "auditoria_actividad.csv"
    )

    # ========================================================
    # TRAIN / TEST
    # ========================================================

    train_df = df[
        df[MONTH_COL].isin(
            TRAIN_MONTHS
        )
    ].copy()

    test_df = df[
        df[MONTH_COL]
        .eq(TEST_MONTH)
    ].copy()

    print()
    print(
        f"Train meses: "
        f"{TRAIN_MONTHS}"
    )

    print(
        f"Test mes   : "
        f"{TEST_MONTH}"
    )

    # ========================================================
    # ABLATION
    # ========================================================

    experiments = {
        "FULL457_BASE":
            features_full,

        "FULL457_PLUS_ACTUAL":
            features_full
            + [
                "actividad_canales",
            ],

        "FULL457_PLUS_MIN3":
            features_full
            + [
                "actividad_min3",
            ],

        "FULL457_PLUS_ACTUAL_MIN3":
            features_full
            + [
                "actividad_canales",
                "actividad_min3",
            ],
    }

    summaries_list = []
    curves_list = []
    importances_list = []
    predictions = pd.DataFrame(
        {
            ID_COL:
                test_df[
                    ID_COL
                ].to_numpy(),
            MONTH_COL:
                test_df[
                    MONTH_COL
                ].to_numpy(),
            TARGET_COL:
                test_df[
                    TARGET_COL
                ].to_numpy(),
        }
    )

    for (
        model_name,
        model_features,
    ) in experiments.items():

        (
            summary,
            curve,
            probabilities,
            importance,
        ) = evaluate_model(
            model_name,
            train_df,
            test_df,
            model_features,
        )

        summaries_list.append(
            summary
        )

        curve = curve.copy()

        curve["model"] = (
            model_name
        )

        curves_list.append(
            curve
        )

        importance = (
            importance.copy()
        )

        importance["model"] = (
            model_name
        )

        importances_list.append(
            importance
        )

        predictions[
            f"pred_{model_name}"
        ] = probabilities

    # ========================================================
    # RESULTADOS
    # ========================================================

    summaries = pd.DataFrame(
        summaries_list
    )

    curves = pd.concat(
        curves_list,
        ignore_index=True,
    )

    importances = pd.concat(
        importances_list,
        ignore_index=True,
    )

    summaries.to_csv(
        OUTPUT_DIR
        / "resumen_modelos.csv",
        index=False,
    )

    curves.to_csv(
        OUTPUT_DIR
        / "curvas_ganancia.csv",
        index=False,
    )

    importances.to_csv(
        OUTPUT_DIR
        / "importancias.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "predicciones_test.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Comparación contra baseline por cutoff
    # --------------------------------------------------------

    base_curve = (
        curves[
            curves["model"]
            == "FULL457_BASE"
        ][
            [
                "cut",
                "gain_millions",
                "positives",
            ]
        ]
        .rename(
            columns={
                "gain_millions":
                    "gain_base",
                "positives":
                    "positives_base",
            }
        )
    )

    comparisons = []

    for model_name in experiments:

        if (
            model_name
            == "FULL457_BASE"
        ):
            continue

        candidate = (
            curves[
                curves["model"]
                == model_name
            ][
                [
                    "cut",
                    "gain_millions",
                    "positives",
                ]
            ]
            .rename(
                columns={
                    "gain_millions":
                        "gain_candidate",
                    "positives":
                        "positives_candidate",
                }
            )
        )

        comp = base_curve.merge(
            candidate,
            on="cut",
            how="inner",
        )

        comp["model"] = (
            model_name
        )

        comp[
            "delta_gain_millions"
        ] = (
            comp[
                "gain_candidate"
            ]
            - comp[
                "gain_base"
            ]
        )

        comp[
            "delta_positives"
        ] = (
            comp[
                "positives_candidate"
            ]
            - comp[
                "positives_base"
            ]
        )

        comparisons.append(
            comp
        )

    comparison = pd.concat(
        comparisons,
        ignore_index=True,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "comparacion_vs_base.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Importancia de las dos features estudiadas
    # --------------------------------------------------------

    imp_activity = (
        importances[
            importances[
                "feature"
            ].isin(
                [
                    "actividad_canales",
                    "actividad_min3",
                ]
            )
        ]
        .copy()
    )

    imp_activity.to_csv(
        OUTPUT_DIR
        / "importancia_actividad.csv",
        index=False,
    )

    # ========================================================
    # IMPRESION
    # ========================================================

    print()
    print("=" * 80)
    print(
        "RESUMEN ABLATION"
    )
    print("=" * 80)

    print(
        summaries.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "COMPARACION VS BASE"
    )
    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "TOP RANKING: CUTS 8000-9500"
    )
    print("=" * 80)

    top_comparison = (
        comparison[
            comparison["cut"]
            .between(
                8000,
                9500,
            )
        ]
        .sort_values(
            [
                "cut",
                "delta_gain_millions",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    print(
        top_comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print(
        "IMPORTANCIA ACTIVIDAD"
    )
    print("=" * 80)

    print(
        imp_activity.to_string(
            index=False
        )
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata_out = {
        "dataset":
            str(DATASET),
        "train_months":
            TRAIN_MONTHS,
        "test_month":
            TEST_MONTH,
        "seed":
            SEED,
        "baseline_features":
            len(features_full),
        "experiments": {
            "FULL457_BASE": [],
            "FULL457_PLUS_ACTUAL": ["actividad_canales"],
            "FULL457_PLUS_MIN3": ["actividad_min3"],
            "FULL457_PLUS_ACTUAL_MIN3": [
                "actividad_canales",
                "actividad_min3"
            ],
        },
        "transaction_variables":
            TRANSACTION_VARS,
        "uses_target_for_features":
            False,
        "uses_202108_for_training":
            False,
    }

    with open(
        OUTPUT_DIR
        / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata_out,
            f,
            indent=2,
            ensure_ascii=False,
        )

    del df
    del train_df
    del test_df

    gc.collect()

    print()
    print("=" * 80)
    print(
        "Z541 FINALIZADO"
    )
    print("=" * 80)

    print(
        f"Resultados: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
