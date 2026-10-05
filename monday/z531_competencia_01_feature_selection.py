#!/usr/bin/env python3
"""
z531_competencia_01_feature_selection.py

Feature selection temporalmente correcta sobre el modelo histórico FULL.

Objetivo
--------
Evaluar si podemos reducir las 457 features de FULL manteniendo
las tres representaciones temporales:

    ORIGINAL + LAG1 + DELTA_LAG1 + lag1_disponible

sin utilizar información del holdout para seleccionar variables.

Ventanas:
    A) train 202104          -> test 202105
    B) train 202104-202105   -> test 202106

Por cada ventana y seed:
1. Entrenar FULL exclusivamente con train.
2. Obtener importancia LightGBM por gain.
3. Construir rankings usando SOLO train.
4. Evaluar:
       FULL
       TOP200
       TOP150
       TOP100
       TOP50
5. Medir AUC, AP, LogLoss y ganancia económica.
6. Evaluar N=12000 como criterio principal.
7. Registrar best cut sólo como diagnóstico.

NO utiliza 202108.
NO genera submits.
NO consulta Public Leaderboard.
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
    "competencia_01/feature_selection_z531"
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

TOP_KS = [
    200,
    150,
    100,
    50,
]

N_PRINCIPAL = 12000

CORTES = list(
    range(
        4000,
        19001,
        500,
    )
)

GANANCIA_ACIERTO = 1_072_500
COSTO_ERROR = -27_500

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
        if c.endswith(
            "_delta_lag1"
        )
    ]

    lag = [
        c
        for c in df.columns
        if (
            c.endswith("_lag1")
            and not c.endswith(
                "_delta_lag1"
            )
        )
    ]

    originales = [
        c
        for c in df.columns
        if (
            c not in excluir
            and not c.endswith(
                "_lag1"
            )
            and not c.endswith(
                "_delta_lag1"
            )
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

    if feature.endswith(
        "_delta_lag1"
    ):
        return "DELTA"

    if feature.endswith(
        "_lag1"
    ):
        return "LAG"

    return "ORIGINAL"


# ============================================================
# Métricas
# ============================================================

def metricas_clasificacion(
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


def evaluar_corte(
    y_true: np.ndarray,
    probs: np.ndarray,
    n: int,
) -> dict:

    n = min(
        n,
        len(probs),
    )

    orden = np.argsort(
        -probs,
        kind="stable",
    )

    seleccion = orden[:n]

    positivos = int(
        y_true[
            seleccion
        ].sum()
    )

    negativos = int(
        n - positivos
    )

    ganancia = (
        positivos
        * GANANCIA_ACIERTO
        + negativos
        * COSTO_ERROR
    )

    precision = (
        positivos / n
        if n > 0
        else np.nan
    )

    total_positivos = int(
        y_true.sum()
    )

    recall = (
        positivos
        / total_positivos
        if total_positivos > 0
        else np.nan
    )

    return {
        "n": n,
        "positivos": positivos,
        "negativos": negativos,
        "precision": precision,
        "recall": recall,
        "ganancia": int(
            ganancia
        ),
    }


def curva_ganancia(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> pd.DataFrame:

    rows = []

    for n in CORTES:

        r = evaluar_corte(
            y_true=y_true,
            probs=probs,
            n=n,
        )

        rows.append(r)

    return pd.DataFrame(
        rows
    )


# ============================================================
# Selección
# ============================================================

def ranking_gain(
    modelo: LGBMClassifier,
    features: list[str],
) -> pd.DataFrame:

    gains = np.asarray(
        modelo.feature_importances_,
        dtype=np.float64,
    )

    if len(gains) != len(
        features
    ):
        raise ValueError(
            "Cantidad de importancias "
            "inconsistente."
        )

    ranking = pd.DataFrame(
        {
            "feature": features,
            "gain": gains,
        }
    )

    ranking[
        "familia"
    ] = ranking[
        "feature"
    ].map(
        familia_feature
    )

    # Para empates usamos nombre de feature,
    # de modo que el ranking sea reproducible.
    ranking = (
        ranking.sort_values(
            [
                "gain",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    ranking[
        "ranking"
    ] = np.arange(
        1,
        len(ranking) + 1,
    )

    total_gain = float(
        ranking["gain"].sum()
    )

    if total_gain > 0:

        ranking[
            "gain_pct"
        ] = (
            100.0
            * ranking["gain"]
            / total_gain
        )

    else:

        ranking[
            "gain_pct"
        ] = 0.0

    return ranking


# ============================================================
# Main
# ============================================================

def main():

    inicio = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80,
        flush=True,
    )
    print(
        "z531 - FEATURE SELECTION HISTÓRICA",
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
        f"\nOutput:\n  {OUTPUT_DIR}",
        flush=True,
    )

    print(
        f"\nSeeds: {SEEDS}",
        flush=True,
    )

    print(
        f"Top-K: {TOP_KS}",
        flush=True,
    )

    print(
        f"N principal: {N_PRINCIPAL:,}",
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

    df = (
        df.loc[
            df[
                MONTH_COL
            ].isin(
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
        f"\nFilas cargadas : "
        f"{len(df):,}",
        flush=True,
    )
    print(
        f"Originales     : "
        f"{len(features_originales)}",
        flush=True,
    )
    print(
        f"Lag1           : "
        f"{len(features_lag)}",
        flush=True,
    )
    print(
        f"Delta lag1     : "
        f"{len(features_delta)}",
        flush=True,
    )
    print(
        f"FULL           : "
        f"{len(features_full)}",
        flush=True,
    )

    if len(
        features_originales
    ) != 152:
        raise ValueError(
            "Se esperaban 152 originales."
        )

    if len(
        features_lag
    ) != 152:
        raise ValueError(
            "Se esperaban 152 lag1."
        )

    if len(
        features_delta
    ) != 152:
        raise ValueError(
            "Se esperaban 152 delta."
        )

    if len(
        features_full
    ) != 457:
        raise ValueError(
            "FULL debe tener 457 features."
        )

    if len(
        set(features_full)
    ) != 457:
        raise ValueError(
            "Hay features duplicadas."
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
        f"Duplicados cliente/mes: "
        f"{duplicados}",
        flush=True,
    )

    if duplicados != 0:
        raise ValueError(
            "Hay duplicados cliente/mes."
        )

    # ========================================================
    # Resultados
    # ========================================================

    resultados = []
    rankings = []
    curvas = []
    composiciones = []

    # ========================================================
    # Ventanas
    # ========================================================

    for ventana in VENTANAS:

        nombre = ventana[
            "nombre"
        ]

        train_months = ventana[
            "train_months"
        ]

        test_month = ventana[
            "test_month"
        ]

        print(
            "\n\n"
            + "#"
            * 80,
            flush=True,
        )
        print(
            f"VENTANA {nombre}",
            flush=True,
        )
        print(
            f"Train: "
            f"{train_months}",
            flush=True,
        )
        print(
            f"Test : "
            f"{test_month}",
            flush=True,
        )
        print(
            "#"
            * 80,
            flush=True,
        )

        train = (
            df.loc[
                df[
                    MONTH_COL
                ].isin(
                    train_months
                )
            ]
            .copy()
        )

        test = (
            df.loc[
                df[
                    MONTH_COL
                ].eq(
                    test_month
                )
            ]
            .copy()
        )

        if train[
            TARGET_COL
        ].isna().any():
            raise ValueError(
                f"{nombre}: "
                "target NA en train."
            )

        if test[
            TARGET_COL
        ].isna().any():
            raise ValueError(
                f"{nombre}: "
                "target NA en test."
            )

        y_train = (
            train[
                TARGET_COL
            ]
            .eq("BAJA+2")
            .astype(
                np.int8
            )
            .to_numpy()
        )

        y_test = (
            test[
                TARGET_COL
            ]
            .eq("BAJA+2")
            .astype(
                np.int8
            )
            .to_numpy()
        )

        print(
            f"\nTrain rows   : "
            f"{len(train):,}",
            flush=True,
        )
        print(
            f"Train BAJA+2 : "
            f"{int(y_train.sum()):,}",
            flush=True,
        )
        print(
            f"Test rows    : "
            f"{len(test):,}",
            flush=True,
        )
        print(
            f"Test BAJA+2  : "
            f"{int(y_test.sum()):,}",
            flush=True,
        )

        # ====================================================
        # Seeds
        # ====================================================

        for seed_idx, seed in enumerate(
            SEEDS,
            start=1,
        ):

            print(
                "\n"
                + "-"
                * 80,
                flush=True,
            )
            print(
                f"Seed "
                f"{seed_idx}/"
                f"{len(SEEDS)}: "
                f"{seed}",
                flush=True,
            )
            print(
                "-"
                * 80,
                flush=True,
            )

            params = (
                PARAMS_BASE.copy()
            )

            params[
                "random_state"
            ] = seed

            # ================================================
            # 1. FULL
            # ================================================

            X_train_full = train[
                features_full
            ]

            X_test_full = test[
                features_full
            ]

            modelo_full = (
                LGBMClassifier(
                    **params
                )
            )

            t0 = time.time()

            modelo_full.fit(
                X_train_full,
                y_train,
            )

            probs_full = (
                modelo_full.predict_proba(
                    X_test_full
                )[:, 1]
            )

            elapsed_full = (
                time.time()
                - t0
            )

            ranking = ranking_gain(
                modelo=modelo_full,
                features=features_full,
            )

            ranking[
                "ventana"
            ] = nombre

            ranking[
                "seed"
            ] = seed

            rankings.append(
                ranking
            )

            # Guardamos ranking individual
            ranking.to_csv(
                OUTPUT_DIR
                / (
                    f"ranking_train_"
                    f"{nombre}_"
                    f"seed{seed}.csv"
                ),
                index=False,
            )

            # ================================================
            # Función local para registrar resultado
            # ================================================

            def registrar(
                modelo_nombre: str,
                probs: np.ndarray,
                n_features: int,
                features_usadas: list[str],
                elapsed: float,
            ) -> None:

                mets = (
                    metricas_clasificacion(
                        y_true=y_test,
                        probs=probs,
                    )
                )

                principal = (
                    evaluar_corte(
                        y_true=y_test,
                        probs=probs,
                        n=N_PRINCIPAL,
                    )
                )

                curva = (
                    curva_ganancia(
                        y_true=y_test,
                        probs=probs,
                    )
                )

                idx_best = (
                    curva[
                        "ganancia"
                    ].idxmax()
                )

                best = (
                    curva.loc[
                        idx_best
                    ]
                )

                resultados.append(
                    {
                        "ventana":
                            nombre,
                        "train_months":
                            ",".join(
                                str(x)
                                for x
                                in train_months
                            ),
                        "test_month":
                            test_month,
                        "seed":
                            seed,
                        "modelo":
                            modelo_nombre,
                        "n_features":
                            n_features,
                        "auc":
                            mets["auc"],
                        "average_precision":
                            mets[
                                "average_precision"
                            ],
                        "logloss":
                            mets[
                                "logloss"
                            ],
                        "n_principal":
                            N_PRINCIPAL,
                        "positivos_n12000":
                            principal[
                                "positivos"
                            ],
                        "precision_n12000":
                            principal[
                                "precision"
                            ],
                        "recall_n12000":
                            principal[
                                "recall"
                            ],
                        "ganancia_n12000":
                            principal[
                                "ganancia"
                            ],
                        "best_n":
                            int(
                                best["n"]
                            ),
                        "best_positivos":
                            int(
                                best[
                                    "positivos"
                                ]
                            ),
                        "best_ganancia":
                            int(
                                best[
                                    "ganancia"
                                ]
                            ),
                        "elapsed_seconds":
                            elapsed,
                    }
                )

                curva[
                    "ventana"
                ] = nombre

                curva[
                    "seed"
                ] = seed

                curva[
                    "modelo"
                ] = modelo_nombre

                curvas.append(
                    curva
                )

                familias = (
                    pd.Series(
                        [
                            familia_feature(
                                f
                            )
                            for f
                            in features_usadas
                        ]
                    )
                    .value_counts()
                )

                composiciones.append(
                    {
                        "ventana":
                            nombre,
                        "seed":
                            seed,
                        "modelo":
                            modelo_nombre,
                        "n_features":
                            n_features,
                        "originales":
                            int(
                                familias.get(
                                    "ORIGINAL",
                                    0,
                                )
                            ),
                        "lag":
                            int(
                                familias.get(
                                    "LAG",
                                    0,
                                )
                            ),
                        "delta":
                            int(
                                familias.get(
                                    "DELTA",
                                    0,
                                )
                            ),
                        "indicador":
                            int(
                                familias.get(
                                    "INDICADOR",
                                    0,
                                )
                            ),
                    }
                )

                print(
                    f"{modelo_nombre:<7} "
                    f"| vars="
                    f"{n_features:>3} "
                    f"| AUC="
                    f"{mets['auc']:.6f} "
                    f"| AP="
                    f"{mets['average_precision']:.6f} "
                    f"| N12000="
                    f"{principal['ganancia'] / 1e6:.3f}M "
                    f"| best N="
                    f"{int(best['n']):>5} "
                    f"gain="
                    f"{int(best['ganancia']) / 1e6:.3f}M "
                    f"| {elapsed:.1f}s",
                    flush=True,
                )

            # FULL
            registrar(
                modelo_nombre="FULL",
                probs=probs_full,
                n_features=len(
                    features_full
                ),
                features_usadas=
                    features_full,
                elapsed=
                    elapsed_full,
            )

            # ================================================
            # 2. TOP-K
            # ================================================

            for k in TOP_KS:

                features_k = (
                    ranking[
                        "feature"
                    ]
                    .head(k)
                    .tolist()
                )

                X_train_k = train[
                    features_k
                ]

                X_test_k = test[
                    features_k
                ]

                modelo_k = (
                    LGBMClassifier(
                        **params
                    )
                )

                t1 = time.time()

                modelo_k.fit(
                    X_train_k,
                    y_train,
                )

                probs_k = (
                    modelo_k.predict_proba(
                        X_test_k
                    )[:, 1]
                )

                elapsed_k = (
                    time.time()
                    - t1
                )

                registrar(
                    modelo_nombre=
                        f"TOP{k}",
                    probs=probs_k,
                    n_features=k,
                    features_usadas=
                        features_k,
                    elapsed=
                        elapsed_k,
                )

    # ========================================================
    # DataFrames
    # ========================================================

    resultados_df = (
        pd.DataFrame(
            resultados
        )
    )

    rankings_df = (
        pd.concat(
            rankings,
            ignore_index=True,
        )
    )

    curvas_df = (
        pd.concat(
            curvas,
            ignore_index=True,
        )
    )

    composiciones_df = (
        pd.DataFrame(
            composiciones
        )
    )

    resultados_df.to_csv(
        OUTPUT_DIR
        / "resultados_por_seed.csv",
        index=False,
    )

    rankings_df.to_csv(
        OUTPUT_DIR
        / "rankings_train.csv",
        index=False,
    )

    curvas_df.to_csv(
        OUTPUT_DIR
        / "curvas_ganancia.csv",
        index=False,
    )

    composiciones_df.to_csv(
        OUTPUT_DIR
        / "composicion_features.csv",
        index=False,
    )

    # ========================================================
    # Resumen por ventana/modelo
    # ========================================================

    resumen = (
        resultados_df.groupby(
            [
                "ventana",
                "modelo",
                "n_features",
            ],
            as_index=False,
        )
        .agg(
            auc_mean=(
                "auc",
                "mean",
            ),
            auc_std=(
                "auc",
                "std",
            ),
            ap_mean=(
                "average_precision",
                "mean",
            ),
            ap_std=(
                "average_precision",
                "std",
            ),
            logloss_mean=(
                "logloss",
                "mean",
            ),
            gain_n12000_mean=(
                "ganancia_n12000",
                "mean",
            ),
            gain_n12000_std=(
                "ganancia_n12000",
                "std",
            ),
            gain_n12000_min=(
                "ganancia_n12000",
                "min",
            ),
            gain_n12000_max=(
                "ganancia_n12000",
                "max",
            ),
            best_gain_mean=(
                "best_ganancia",
                "mean",
            ),
        )
    )

    orden_modelos = {
        "FULL": 0,
        "TOP200": 1,
        "TOP150": 2,
        "TOP100": 3,
        "TOP50": 4,
    }

    resumen[
        "_orden"
    ] = resumen[
        "modelo"
    ].map(
        orden_modelos
    )

    resumen = (
        resumen.sort_values(
            [
                "ventana",
                "_orden",
            ]
        )
        .drop(
            columns="_orden"
        )
        .reset_index(
            drop=True
        )
    )

    resumen.to_csv(
        OUTPUT_DIR
        / "resumen_por_ventana.csv",
        index=False,
    )

    # ========================================================
    # Deltas contra FULL
    # ========================================================

    comparaciones = []

    for ventana in [
        "202105",
        "202106",
    ]:

        sub = (
            resumen.loc[
                resumen[
                    "ventana"
                ].eq(
                    ventana
                )
            ]
            .copy()
        )

        full_row = (
            sub.loc[
                sub[
                    "modelo"
                ].eq(
                    "FULL"
                )
            ]
            .iloc[0]
        )

        gain_full = float(
            full_row[
                "gain_n12000_mean"
            ]
        )

        for _, row in (
            sub.iterrows()
        ):

            gain = float(
                row[
                    "gain_n12000_mean"
                ]
            )

            delta = (
                gain
                - gain_full
            )

            pct = (
                100.0
                * delta
                / gain_full
                if gain_full != 0
                else np.nan
            )

            comparaciones.append(
                {
                    "ventana":
                        ventana,
                    "modelo":
                        row["modelo"],
                    "n_features":
                        int(
                            row[
                                "n_features"
                            ]
                        ),
                    "gain_n12000_mean":
                        gain,
                    "delta_vs_full":
                        delta,
                    "delta_pct_vs_full":
                        pct,
                }
            )

    comparaciones_df = (
        pd.DataFrame(
            comparaciones
        )
    )

    comparaciones_df.to_csv(
        OUTPUT_DIR
        / "comparacion_vs_full.csv",
        index=False,
    )

    # ========================================================
    # Resumen global
    # ========================================================

    global_rows = []

    for modelo in [
        "FULL",
        "TOP200",
        "TOP150",
        "TOP100",
        "TOP50",
    ]:

        sub = (
            comparaciones_df.loc[
                comparaciones_df[
                    "modelo"
                ].eq(
                    modelo
                )
            ]
        )

        wins = int(
            (
                sub[
                    "delta_vs_full"
                ] > 0
            ).sum()
        )

        ties = int(
            (
                sub[
                    "delta_vs_full"
                ] == 0
            ).sum()
        )

        losses = int(
            (
                sub[
                    "delta_vs_full"
                ] < 0
            ).sum()
        )

        global_rows.append(
            {
                "modelo":
                    modelo,
                "gain_mean":
                    float(
                        sub[
                            "gain_n12000_mean"
                        ].mean()
                    ),
                "delta_vs_full_mean":
                    float(
                        sub[
                            "delta_vs_full"
                        ].mean()
                    ),
                "wins_vs_full":
                    wins,
                "ties_vs_full":
                    ties,
                "losses_vs_full":
                    losses,
            }
        )

    global_df = pd.DataFrame(
        global_rows
    )

    global_df.to_csv(
        OUTPUT_DIR
        / "resumen_global.csv",
        index=False,
    )

    # ========================================================
    # Composición media
    # ========================================================

    composicion_resumen = (
        composiciones_df.groupby(
            [
                "ventana",
                "modelo",
                "n_features",
            ],
            as_index=False,
        )
        .agg(
            originales_mean=(
                "originales",
                "mean",
            ),
            lag_mean=(
                "lag",
                "mean",
            ),
            delta_mean=(
                "delta",
                "mean",
            ),
            indicador_mean=(
                "indicador",
                "mean",
            ),
        )
    )

    composicion_resumen.to_csv(
        OUTPUT_DIR
        / "resumen_composicion.csv",
        index=False,
    )

    # ========================================================
    # Print final
    # ========================================================

    print(
        "\n\n"
        + "="
        * 80,
        flush=True,
    )
    print(
        "RESUMEN POR VENTANA",
        flush=True,
    )
    print(
        "="
        * 80,
        flush=True,
    )

    for ventana in [
        "202105",
        "202106",
    ]:

        print(
            f"\nVENTANA "
            f"{ventana}",
            flush=True,
        )

        sub = (
            resumen.loc[
                resumen[
                    "ventana"
                ].eq(
                    ventana
                )
            ]
        )

        for _, row in (
            sub.iterrows()
        ):

            print(
                f"{row['modelo']:<7} "
                f"| vars="
                f"{int(row['n_features']):>3} "
                f"| AUC="
                f"{row['auc_mean']:.6f} "
                f"± "
                f"{row['auc_std']:.6f} "
                f"| AP="
                f"{row['ap_mean']:.6f} "
                f"| gain N12000="
                f"{row['gain_n12000_mean'] / 1e6:.3f}M "
                f"± "
                f"{row['gain_n12000_std'] / 1e6:.3f}M",
                flush=True,
            )

        print(
            "\n  Delta vs FULL:",
            flush=True,
        )

        comp = (
            comparaciones_df.loc[
                comparaciones_df[
                    "ventana"
                ].eq(
                    ventana
                )
            ]
        )

        for _, row in (
            comp.iterrows()
        ):

            print(
                f"    "
                f"{row['modelo']:<7} "
                f"{row['delta_vs_full'] / 1e6:+.3f}M "
                f"("
                f"{row['delta_pct_vs_full']:+.2f}%"
                f")",
                flush=True,
            )

    print(
        "\n"
        + "="
        * 80,
        flush=True,
    )
    print(
        "RESUMEN GLOBAL",
        flush=True,
    )
    print(
        "="
        * 80,
        flush=True,
    )

    for _, row in (
        global_df.iterrows()
    ):

        print(
            f"{row['modelo']:<7} "
            f"| gain medio="
            f"{row['gain_mean'] / 1e6:.3f}M "
            f"| delta FULL="
            f"{row['delta_vs_full_mean'] / 1e6:+.3f}M "
            f"| W/T/L="
            f"{int(row['wins_vs_full'])}/"
            f"{int(row['ties_vs_full'])}/"
            f"{int(row['losses_vs_full'])}",
            flush=True,
        )

    print(
        "\n"
        + "="
        * 80,
        flush=True,
    )
    print(
        "COMPOSICIÓN MEDIA DE FEATURES",
        flush=True,
    )
    print(
        "="
        * 80,
        flush=True,
    )

    for _, row in (
        composicion_resumen.iterrows()
    ):

        print(
            f"{row['ventana']} "
            f"{row['modelo']:<7} "
            f"| O="
            f"{row['originales_mean']:.1f} "
            f"L="
            f"{row['lag_mean']:.1f} "
            f"D="
            f"{row['delta_mean']:.1f} "
            f"I="
            f"{row['indicador_mean']:.1f}",
            flush=True,
        )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "script":
            "z531_competencia_01_feature_selection.py",

        "dataset":
            str(DATASET),

        "target":
            "BAJA+2",

        "features_full":
            len(features_full),

        "top_ks":
            TOP_KS,

        "n_principal":
            N_PRINCIPAL,

        "cuts":
            CORTES,

        "seeds":
            SEEDS,

        "ventanas":
            VENTANAS,

        "selection_rule":
            (
                "LightGBM gain ranking "
                "computed exclusively "
                "from each training set "
                "and seed"
            ),

        "uses_holdout_for_selection":
            False,

        "uses_202108":
            False,

        "generates_submit":
            False,

        "params":
            PARAMS_BASE,
    }

    with (
        OUTPUT_DIR
        / "metadata_z531.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    elapsed_total = (
        time.time()
        - inicio
    )

    print(
        f"\nTiempo total: "
        f"{elapsed_total / 60:.1f} min",
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