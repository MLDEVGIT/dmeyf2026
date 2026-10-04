#!/usr/bin/env python3
"""
z530_competencia_01_importancia_historicas.py

Diagnóstico de importancia de variables del modelo histórico FULL.

Objetivo
--------
Entender qué variables explican la señal del modelo FULL de z527:

    152 originales
    152 lag1
    152 delta_lag1
      1 lag1_disponible
    ------------------
    457 features

Se utilizan exclusivamente los dos holdouts temporales ya establecidos:

Ventana A
    train = 202104
    test  = 202105

Ventana B
    train = 202104 + 202105
    test  = 202106

Para cada ventana se entrenan las mismas 5 seeds utilizadas en
z525/z527/z529.

IMPORTANTE
----------
Este experimento:
- NO utiliza 202108.
- NO genera archivos de submit.
- NO consulta el Public Leaderboard.
- NO selecciona todavía variables para un modelo final.
- Es únicamente diagnóstico.

La importancia se obtiene mediante LightGBM importance_type="gain".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


# ============================================================
# Configuración
# ============================================================

DATASET = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/feature_engineering_z523/"
    "competencia_01_historico_lag1.parquet"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/importancia_historicas_z530"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
LAG_AVAILABLE_COL = "lag1_disponible"

SEEDS = [
    290497,
    100003,
    200003,
    300007,
    400009,
]

VENTANAS = [
    {
        "nombre": "202105",
        "train_months": [202104],
        "test_month": 202105,
    },
    {
        "nombre": "202106",
        "train_months": [202104, 202105],
        "test_month": 202106,
    },
]

PARAMS_BASE = {
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
    "n_jobs": -1,
    "verbosity": -1,
    "importance_type": "gain",
}


# ============================================================
# Features
# ============================================================

def identificar_features(
    df: pd.DataFrame,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
]:

    excluir = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
        LAG_AVAILABLE_COL,
    }

    delta = [
        c
        for c in df.columns
        if c.endswith("_delta_lag1")
    ]

    lag = [
        c
        for c in df.columns
        if (
            c.endswith("_lag1")
            and not c.endswith("_delta_lag1")
        )
    ]

    originales = [
        c
        for c in df.columns
        if (
            c not in excluir
            and not c.endswith("_lag1")
            and not c.endswith("_delta_lag1")
        )
    ]

    full = (
        originales
        + lag
        + delta
        + [LAG_AVAILABLE_COL]
    )

    return (
        originales,
        lag,
        delta,
        full,
    )


def familia_feature(
    feature: str,
) -> str:

    if feature == LAG_AVAILABLE_COL:
        return "INDICADOR"

    if feature.endswith("_delta_lag1"):
        return "DELTA"

    if feature.endswith("_lag1"):
        return "LAG"

    return "ORIGINAL"


def variable_base(
    feature: str,
) -> str:

    if feature == LAG_AVAILABLE_COL:
        return LAG_AVAILABLE_COL

    if feature.endswith("_delta_lag1"):
        return feature[
            :-len("_delta_lag1")
        ]

    if feature.endswith("_lag1"):
        return feature[
            :-len("_lag1")
        ]

    return feature


# ============================================================
# Métricas
# ============================================================

def calcular_metricas(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict:

    return {
        "auc": float(
            roc_auc_score(
                y_true,
                probs,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                probs,
            )
        ),
        "logloss": float(
            log_loss(
                y_true,
                probs,
                labels=[0, 1],
            )
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    inicio = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80, flush=True)
    print(
        "z530 - IMPORTANCIA DE FEATURES HISTÓRICAS",
        flush=True,
    )
    print("=" * 80, flush=True)

    print(
        f"\nDataset:\n  {DATASET}",
        flush=True,
    )

    print(
        f"\nOutput:\n  {OUTPUT_DIR}",
        flush=True,
    )

    print(
        f"\nSeeds: {SEEDS}",
        flush=True,
    )

    # ========================================================
    # Lectura
    # ========================================================

    print(
        "\nLeyendo parquet...",
        flush=True,
    )

    df = pd.read_parquet(
        DATASET
    )

    # Sólo meses necesarios.
    df = (
        df.loc[
            df[MONTH_COL].isin(
                [
                    202104,
                    202105,
                    202106,
                ]
            )
        ]
        .copy()
    )

    (
        features_originales,
        features_lag,
        features_delta,
        features_full,
    ) = identificar_features(
        df
    )

    print(
        f"\nFilas cargadas     : {len(df):,}",
        flush=True,
    )
    print(
        f"Originales         : {len(features_originales):,}",
        flush=True,
    )
    print(
        f"Lag1               : {len(features_lag):,}",
        flush=True,
    )
    print(
        f"Delta lag1         : {len(features_delta):,}",
        flush=True,
    )
    print(
        f"FULL               : {len(features_full):,}",
        flush=True,
    )

    # ========================================================
    # Auditorías
    # ========================================================

    if len(features_originales) != 152:
        raise ValueError(
            "Se esperaban 152 features originales."
        )

    if len(features_lag) != 152:
        raise ValueError(
            "Se esperaban 152 features lag1."
        )

    if len(features_delta) != 152:
        raise ValueError(
            "Se esperaban 152 features delta_lag1."
        )

    if len(features_full) != 457:
        raise ValueError(
            "FULL debe tener 457 features."
        )

    if len(set(features_full)) != 457:
        raise ValueError(
            "Hay features duplicadas en FULL."
        )

    duplicados = int(
        df.duplicated(
            subset=[
                ID_COL,
                MONTH_COL,
            ]
        ).sum()
    )

    print(
        f"Duplicados cliente/mes: {duplicados}",
        flush=True,
    )

    if duplicados != 0:
        raise ValueError(
            "Hay duplicados cliente/mes."
        )

    # ========================================================
    # Acumuladores
    # ========================================================

    importancias = []
    metricas_modelos = []

    # ========================================================
    # Entrenamiento por ventana
    # ========================================================

    for ventana in VENTANAS:

        nombre = ventana["nombre"]
        train_months = ventana["train_months"]
        test_month = ventana["test_month"]

        print(
            "\n\n" + "#" * 80,
            flush=True,
        )
        print(
            f"VENTANA {nombre}",
            flush=True,
        )
        print(
            f"Train: {train_months}",
            flush=True,
        )
        print(
            f"Test : {test_month}",
            flush=True,
        )
        print(
            "#" * 80,
            flush=True,
        )

        train = (
            df.loc[
                df[MONTH_COL].isin(
                    train_months
                )
            ]
            .copy()
        )

        test = (
            df.loc[
                df[MONTH_COL].eq(
                    test_month
                )
            ]
            .copy()
        )

        if train[TARGET_COL].isna().any():
            raise ValueError(
                f"{nombre}: target NA en train."
            )

        if test[TARGET_COL].isna().any():
            raise ValueError(
                f"{nombre}: target NA en test."
            )

        y_train = (
            train[TARGET_COL]
            .eq("BAJA+2")
            .astype(np.int8)
            .to_numpy()
        )

        y_test = (
            test[TARGET_COL]
            .eq("BAJA+2")
            .astype(np.int8)
            .to_numpy()
        )

        X_train = train[
            features_full
        ]

        X_test = test[
            features_full
        ]

        print(
            f"\nTrain rows   : {len(train):,}",
            flush=True,
        )
        print(
            f"Train BAJA+2 : {int(y_train.sum()):,}",
            flush=True,
        )
        print(
            f"Test rows    : {len(test):,}",
            flush=True,
        )
        print(
            f"Test BAJA+2  : {int(y_test.sum()):,}",
            flush=True,
        )

        for i, seed in enumerate(
            SEEDS,
            start=1,
        ):

            params = PARAMS_BASE.copy()
            params["random_state"] = seed

            modelo = LGBMClassifier(
                **params
            )

            t0 = time.time()

            modelo.fit(
                X_train,
                y_train,
            )

            probs = modelo.predict_proba(
                X_test
            )[:, 1]

            elapsed = (
                time.time() - t0
            )

            metricas = calcular_metricas(
                y_true=y_test,
                probs=probs,
            )

            gains = np.asarray(
                modelo.feature_importances_,
                dtype=np.float64,
            )

            if len(gains) != len(
                features_full
            ):
                raise ValueError(
                    "Cantidad de importancias "
                    "inconsistente."
                )

            gain_total = float(
                gains.sum()
            )

            if gain_total <= 0:
                raise ValueError(
                    "Gain total no positivo."
                )

            metricas_modelos.append(
                {
                    "ventana": nombre,
                    "train_months": ",".join(
                        str(x)
                        for x in train_months
                    ),
                    "test_month": test_month,
                    "seed": seed,
                    "auc": metricas["auc"],
                    "average_precision":
                        metricas[
                            "average_precision"
                        ],
                    "logloss":
                        metricas["logloss"],
                    "gain_total_features":
                        gain_total,
                    "elapsed_seconds":
                        elapsed,
                }
            )

            for feature, gain in zip(
                features_full,
                gains,
            ):

                importancias.append(
                    {
                        "ventana": nombre,
                        "seed": seed,
                        "feature": feature,
                        "familia":
                            familia_feature(
                                feature
                            ),
                        "variable_base":
                            variable_base(
                                feature
                            ),
                        "gain": float(gain),
                        "gain_pct_modelo":
                            float(
                                100.0
                                * gain
                                / gain_total
                            ),
                    }
                )

            print(
                f"[{i}/5] seed={seed} "
                f"| AUC={metricas['auc']:.6f} "
                f"| AP="
                f"{metricas['average_precision']:.6f} "
                f"| LL="
                f"{metricas['logloss']:.6f} "
                f"| {elapsed:.1f}s",
                flush=True,
            )

    # ========================================================
    # DataFrames base
    # ========================================================

    imp_df = pd.DataFrame(
        importancias
    )

    metricas_df = pd.DataFrame(
        metricas_modelos
    )

    imp_df.to_csv(
        OUTPUT_DIR
        / "importancias_por_modelo.csv",
        index=False,
    )

    metricas_df.to_csv(
        OUTPUT_DIR
        / "metricas_modelos.csv",
        index=False,
    )

    # ========================================================
    # Importancia media por feature
    # ========================================================

    resumen_features = (
        imp_df.groupby(
            [
                "feature",
                "familia",
                "variable_base",
            ],
            as_index=False,
        )
        .agg(
            gain_pct_mean=(
                "gain_pct_modelo",
                "mean",
            ),
            gain_pct_std=(
                "gain_pct_modelo",
                "std",
            ),
            gain_pct_min=(
                "gain_pct_modelo",
                "min",
            ),
            gain_pct_max=(
                "gain_pct_modelo",
                "max",
            ),
            modelos=(
                "gain_pct_modelo",
                "size",
            ),
        )
        .sort_values(
            "gain_pct_mean",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    resumen_features[
        "ranking_global"
    ] = np.arange(
        1,
        len(resumen_features) + 1,
    )

    resumen_features.to_csv(
        OUTPUT_DIR
        / "ranking_importancia_global.csv",
        index=False,
    )

    # ========================================================
    # Ranking por ventana
    # ========================================================

    ranking_ventanas = (
        imp_df.groupby(
            [
                "ventana",
                "feature",
                "familia",
                "variable_base",
            ],
            as_index=False,
        )
        .agg(
            gain_pct_mean=(
                "gain_pct_modelo",
                "mean",
            ),
            gain_pct_std=(
                "gain_pct_modelo",
                "std",
            ),
        )
    )

    ranking_ventanas[
        "ranking_ventana"
    ] = (
        ranking_ventanas.groupby(
            "ventana"
        )[
            "gain_pct_mean"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    ranking_ventanas = (
        ranking_ventanas.sort_values(
            [
                "ventana",
                "ranking_ventana",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    ranking_ventanas.to_csv(
        OUTPUT_DIR
        / "ranking_importancia_por_ventana.csv",
        index=False,
    )

    # ========================================================
    # Importancia por familia
    # ========================================================

    familia_modelo = (
        imp_df.groupby(
            [
                "ventana",
                "seed",
                "familia",
            ],
            as_index=False,
        )[
            "gain_pct_modelo"
        ]
        .sum()
    )

    resumen_familias = (
        familia_modelo.groupby(
            "familia",
            as_index=False,
        )
        .agg(
            gain_pct_mean=(
                "gain_pct_modelo",
                "mean",
            ),
            gain_pct_std=(
                "gain_pct_modelo",
                "std",
            ),
            gain_pct_min=(
                "gain_pct_modelo",
                "min",
            ),
            gain_pct_max=(
                "gain_pct_modelo",
                "max",
            ),
        )
        .sort_values(
            "gain_pct_mean",
            ascending=False,
        )
    )

    familia_modelo.to_csv(
        OUTPUT_DIR
        / "importancia_familia_por_modelo.csv",
        index=False,
    )

    resumen_familias.to_csv(
        OUTPUT_DIR
        / "resumen_importancia_familias.csv",
        index=False,
    )

    # ========================================================
    # Familia por ventana
    # ========================================================

    familia_ventana = (
        familia_modelo.groupby(
            [
                "ventana",
                "familia",
            ],
            as_index=False,
        )
        .agg(
            gain_pct_mean=(
                "gain_pct_modelo",
                "mean",
            ),
            gain_pct_std=(
                "gain_pct_modelo",
                "std",
            ),
        )
    )

    familia_ventana.to_csv(
        OUTPUT_DIR
        / "importancia_familia_por_ventana.csv",
        index=False,
    )

    # ========================================================
    # Consistencia Top-K
    # ========================================================

    topks = [
        20,
        50,
        100,
        150,
        200,
    ]

    consistencia_rows = []

    for k in topks:

        top_a = set(
            ranking_ventanas.loc[
                (
                    ranking_ventanas[
                        "ventana"
                    ].eq("202105")
                )
                & (
                    ranking_ventanas[
                        "ranking_ventana"
                    ] <= k
                ),
                "feature",
            ]
        )

        top_b = set(
            ranking_ventanas.loc[
                (
                    ranking_ventanas[
                        "ventana"
                    ].eq("202106")
                )
                & (
                    ranking_ventanas[
                        "ranking_ventana"
                    ] <= k
                ),
                "feature",
            ]
        )

        inter = top_a & top_b
        union = top_a | top_b

        jac = (
            len(inter) / len(union)
            if union
            else 1.0
        )

        consistencia_rows.append(
            {
                "top_k": k,
                "interseccion":
                    len(inter),
                "jaccard":
                    jac,
            }
        )

    consistencia_df = pd.DataFrame(
        consistencia_rows
    )

    consistencia_df.to_csv(
        OUTPUT_DIR
        / "consistencia_topk_ventanas.csv",
        index=False,
    )

    # ========================================================
    # Importancia agregada por variable base
    # ========================================================

    # Acá sumamos ORIGINAL + LAG + DELTA de una misma
    # variable conceptual. Sirve para detectar qué variables
    # son importantes aunque la señal aparezca en diferentes
    # representaciones temporales.
    base_modelo = (
        imp_df.loc[
            ~imp_df[
                "familia"
            ].eq("INDICADOR")
        ]
        .groupby(
            [
                "ventana",
                "seed",
                "variable_base",
            ],
            as_index=False,
        )[
            "gain_pct_modelo"
        ]
        .sum()
    )

    resumen_base = (
        base_modelo.groupby(
            "variable_base",
            as_index=False,
        )
        .agg(
            gain_pct_mean=(
                "gain_pct_modelo",
                "mean",
            ),
            gain_pct_std=(
                "gain_pct_modelo",
                "std",
            ),
            gain_pct_min=(
                "gain_pct_modelo",
                "min",
            ),
            gain_pct_max=(
                "gain_pct_modelo",
                "max",
            ),
        )
        .sort_values(
            "gain_pct_mean",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    resumen_base[
        "ranking"
    ] = np.arange(
        1,
        len(resumen_base) + 1,
    )

    resumen_base.to_csv(
        OUTPUT_DIR
        / "ranking_variables_base.csv",
        index=False,
    )

    # ========================================================
    # Salida
    # ========================================================

    print(
        "\n\n" + "=" * 80,
        flush=True,
    )
    print(
        "IMPORTANCIA MEDIA POR FAMILIA",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    for _, row in (
        resumen_familias.iterrows()
    ):

        print(
            f"{row['familia']:<10} "
            f"| gain="
            f"{row['gain_pct_mean']:6.2f}% "
            f"± "
            f"{row['gain_pct_std']:5.2f}",
            flush=True,
        )

    print(
        "\nPor ventana:",
        flush=True,
    )

    for ventana in [
        "202105",
        "202106",
    ]:

        print(
            f"\n  {ventana}",
            flush=True,
        )

        sub = (
            familia_ventana.loc[
                familia_ventana[
                    "ventana"
                ].eq(
                    ventana
                )
            ]
            .sort_values(
                "gain_pct_mean",
                ascending=False,
            )
        )

        for _, row in sub.iterrows():

            print(
                f"    "
                f"{row['familia']:<10} "
                f"{row['gain_pct_mean']:6.2f}% "
                f"± "
                f"{row['gain_pct_std']:5.2f}",
                flush=True,
            )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )
    print(
        "TOP 30 FEATURES GLOBALES",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    for _, row in (
        resumen_features.head(
            30
        ).iterrows()
    ):

        print(
            f"{int(row['ranking_global']):>2}. "
            f"{row['feature']:<45} "
            f"| {row['familia']:<8} "
            f"| gain="
            f"{row['gain_pct_mean']:.4f}% "
            f"± "
            f"{row['gain_pct_std']:.4f}",
            flush=True,
        )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )
    print(
        "TOP 20 VARIABLES BASE",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    for _, row in (
        resumen_base.head(
            20
        ).iterrows()
    ):

        print(
            f"{int(row['ranking']):>2}. "
            f"{row['variable_base']:<45} "
            f"| gain agregado="
            f"{row['gain_pct_mean']:.4f}% "
            f"± "
            f"{row['gain_pct_std']:.4f}",
            flush=True,
        )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )
    print(
        "CONSISTENCIA ENTRE HOLDOUTS",
        flush=True,
    )
    print(
        "=" * 80,
        flush=True,
    )

    for _, row in (
        consistencia_df.iterrows()
    ):

        print(
            f"Top {int(row['top_k']):>3}: "
            f"intersección="
            f"{int(row['interseccion']):>3} "
            f"| Jaccard="
            f"{row['jaccard']:.4f}",
            flush=True,
        )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "script":
            "z530_competencia_01_importancia_historicas.py",

        "dataset":
            str(DATASET),

        "target":
            "BAJA+2",

        "features_total":
            len(features_full),

        "features_originales":
            len(features_originales),

        "features_lag1":
            len(features_lag),

        "features_delta_lag1":
            len(features_delta),

        "indicador":
            LAG_AVAILABLE_COL,

        "seeds":
            SEEDS,

        "ventanas":
            VENTANAS,

        "params":
            PARAMS_BASE,

        "uses_202108":
            False,

        "generates_submit":
            False,
    }

    with (
        OUTPUT_DIR
        / "metadata_z530.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    elapsed = (
        time.time() - inicio
    )

    print(
        f"\nTiempo total: "
        f"{elapsed / 60:.1f} min",
        flush=True,
    )

    print(
        f"\nResultados:\n  "
        f"{OUTPUT_DIR}",
        flush=True,
    )

    print(
        "\nEste experimento NO utilizó "
        "202108 y NO generó submits.",
        flush=True,
    )


if __name__ == "__main__":
    main()