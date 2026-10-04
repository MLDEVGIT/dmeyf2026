#!/usr/bin/env python3
"""
z528_competencia_01_ablation_feature_engineering.py

Ablation study del Feature Engineering histórico para Competencia 1.

Objetivo
--------
Determinar qué componente temporal aporta la mejora observada en z524-z527:

1. BASE
   152 variables originales

2. LAG
   152 originales
 + 152 lag1
 + lag1_disponible
 = 305 features

3. DELTA
   152 originales
 + 152 delta_lag1
 + lag1_disponible
 = 305 features

4. FULL
   152 originales
 + 152 lag1
 + 152 delta_lag1
 + lag1_disponible
 = 457 features

Validación temporal
-------------------
Ventana A:
    train = 202104
    test  = 202105

Ventana B:
    train = 202104 + 202105
    test  = 202106

Todo lo demás queda fijo:
- target BAJA+2
- mismos hiperparámetros LightGBM
- mismas 5 seeds
- N principal = 12000
- misma función de ganancia
- ensemble = promedio de probabilidades

NO utiliza:
- 202107 como target
- 202108
- Public Leaderboard

El objetivo es entender si la señal histórica proviene principalmente de:
- nivel previo (lag1),
- cambio respecto del mes anterior (delta_lag1),
- o complementariedad entre ambos.
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
    "competencia_01/ablation_feature_engineering_z528"
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

N_FIJO = 12000

CUTS = list(
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
}

VENTANAS = [
    {
        "nombre": "202105",
        "train_months": [202104],
        "test_month": 202105,
    },
    {
        "nombre": "202106",
        "train_months": [
            202104,
            202105,
        ],
        "test_month": 202106,
    },
]


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


def curva_ganancia(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> pd.DataFrame:

    order = np.argsort(
        -probs,
        kind="mergesort",
    )

    total_positivos = int(
        y_true.sum()
    )

    filas = []

    for n in CUTS:

        if n > len(y_true):
            continue

        idx = order[:n]

        positivos = int(
            y_true[idx].sum()
        )

        negativos = (
            n - positivos
        )

        gain = (
            positivos
            * GANANCIA_ACIERTO
            + negativos
            * COSTO_ERROR
        )

        filas.append(
            {
                "n": n,
                "gain_millones":
                    float(
                        gain / 1_000_000
                    ),
                "positivos":
                    positivos,
                "precision":
                    float(
                        positivos / n
                    ),
                "recall":
                    float(
                        positivos
                        / total_positivos
                    ),
            }
        )

    return pd.DataFrame(
        filas
    )


def evaluar(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict:

    metricas = calcular_metricas(
        y_true,
        probs,
    )

    curva = curva_ganancia(
        y_true,
        probs,
    )

    mejor = curva.loc[
        curva[
            "gain_millones"
        ].idxmax()
    ]

    fijo = curva.loc[
        curva[
            "n"
        ].eq(
            N_FIJO
        )
    ].iloc[0]

    return {
        "auc":
            metricas["auc"],

        "average_precision":
            metricas[
                "average_precision"
            ],

        "logloss":
            metricas["logloss"],

        "gain_n12000":
            float(
                fijo[
                    "gain_millones"
                ]
            ),

        "positivos_n12000":
            int(
                fijo[
                    "positivos"
                ]
            ),

        "precision_n12000":
            float(
                fijo[
                    "precision"
                ]
            ),

        "recall_n12000":
            float(
                fijo[
                    "recall"
                ]
            ),

        "best_n":
            int(
                mejor["n"]
            ),

        "best_gain":
            float(
                mejor[
                    "gain_millones"
                ]
            ),
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
            c.endswith(
                "_lag1"
            )
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

    return (
        originales,
        lag,
        delta,
    )


# ============================================================
# Jaccard
# ============================================================

def top_n_ids(
    ids: np.ndarray,
    probs: np.ndarray,
) -> set[int]:

    order = np.lexsort(
        (
            ids,
            -probs,
        )
    )

    return set(
        ids[
            order[:N_FIJO]
        ].tolist()
    )


def jaccard(
    a: set[int],
    b: set[int],
) -> float:

    union = a | b

    if not union:
        return 1.0

    return (
        len(a & b)
        / len(union)
    )


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
        "z528 - ABLATION FEATURE ENGINEERING HISTÓRICO",
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
        f"Corte fijo: N={N_FIJO:,}",
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
    ) = identificar_features(
        df
    )

    configuraciones = {
        "BASE": (
            features_originales
        ),

        "LAG": (
            features_originales
            + features_lag
            + [
                LAG_AVAILABLE_COL
            ]
        ),

        "DELTA": (
            features_originales
            + features_delta
            + [
                LAG_AVAILABLE_COL
            ]
        ),

        "FULL": (
            features_originales
            + features_lag
            + features_delta
            + [
                LAG_AVAILABLE_COL
            ]
        ),
    }

    print(
        f"\nFilas cargadas      : "
        f"{len(df):,}",
        flush=True,
    )

    print(
        f"Originales          : "
        f"{len(features_originales):,}",
        flush=True,
    )

    print(
        f"Lag1                : "
        f"{len(features_lag):,}",
        flush=True,
    )

    print(
        f"Delta lag1          : "
        f"{len(features_delta):,}",
        flush=True,
    )

    print(
        "\nConfiguraciones:",
        flush=True,
    )

    for nombre, features in (
        configuraciones.items()
    ):

        print(
            f"  {nombre:<6}: "
            f"{len(features):,} features",
            flush=True,
        )

    # ========================================================
    # Auditorías
    # ========================================================

    if len(
        features_originales
    ) != 152:

        raise ValueError(
            "Se esperaban 152 "
            "features originales."
        )

    if len(
        features_lag
    ) != 152:

        raise ValueError(
            "Se esperaban 152 "
            "features lag1."
        )

    if len(
        features_delta
    ) != 152:

        raise ValueError(
            "Se esperaban 152 "
            "features delta_lag1."
        )

    if len(
        configuraciones[
            "BASE"
        ]
    ) != 152:

        raise ValueError(
            "BASE debe tener 152 features."
        )

    if len(
        configuraciones[
            "LAG"
        ]
    ) != 305:

        raise ValueError(
            "LAG debe tener 305 features."
        )

    if len(
        configuraciones[
            "DELTA"
        ]
    ) != 305:

        raise ValueError(
            "DELTA debe tener 305 features."
        )

    if len(
        configuraciones[
            "FULL"
        ]
    ) != 457:

        raise ValueError(
            "FULL debe tener 457 features."
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
        f"\nDuplicados cliente/mes: "
        f"{duplicados}",
        flush=True,
    )

    if duplicados != 0:

        raise ValueError(
            "Hay duplicados cliente/mes."
        )

    # ========================================================
    # Resultados globales
    # ========================================================

    resultados_seeds = []
    resultados_ensembles = []

    # ========================================================
    # Ventanas
    # ========================================================

    for ventana in VENTANAS:

        nombre_ventana = (
            ventana["nombre"]
        )

        train_months = (
            ventana[
                "train_months"
            ]
        )

        test_month = (
            ventana[
                "test_month"
            ]
        )

        print(
            "\n\n"
            + "#" * 80,
            flush=True,
        )

        print(
            f"VENTANA {nombre_ventana}",
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

        if (
            train[
                TARGET_COL
            ]
            .isna()
            .any()
        ):

            raise ValueError(
                "Target NA en train."
            )

        if (
            test[
                TARGET_COL
            ]
            .isna()
            .any()
        ):

            raise ValueError(
                "Target NA en test."
            )

        y_train = (
            train[
                TARGET_COL
            ]
            .eq(
                "BAJA+2"
            )
            .astype(
                np.int8
            )
            .to_numpy()
        )

        y_test = (
            test[
                TARGET_COL
            ]
            .eq(
                "BAJA+2"
            )
            .astype(
                np.int8
            )
            .to_numpy()
        )

        ids_test = (
            test[
                ID_COL
            ]
            .astype(
                np.int64
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

        # Guardamos ensemble por configuración
        # para luego comparar rankings.
        ensembles_ventana = {}

        for config_nombre, features in (
            configuraciones.items()
        ):

            print(
                "\n"
                + "=" * 80,
                flush=True,
            )

            print(
                f"{nombre_ventana} - "
                f"{config_nombre} "
                f"({len(features)} features)",
                flush=True,
            )

            print(
                "=" * 80,
                flush=True,
            )

            X_train = train[
                features
            ]

            X_test = test[
                features
            ]

            probs_seeds = []
            gains_seeds = []

            for i, seed in enumerate(
                SEEDS,
                start=1,
            ):

                params = (
                    PARAMS_BASE.copy()
                )

                params[
                    "random_state"
                ] = seed

                modelo = (
                    LGBMClassifier(
                        **params
                    )
                )

                t0 = time.time()

                modelo.fit(
                    X_train,
                    y_train,
                )

                probs = (
                    modelo.predict_proba(
                        X_test
                    )[:, 1]
                )

                elapsed = (
                    time.time()
                    - t0
                )

                evaluacion = evaluar(
                    y_true=y_test,
                    probs=probs,
                )

                probs_seeds.append(
                    probs
                )

                gains_seeds.append(
                    evaluacion[
                        "gain_n12000"
                    ]
                )

                resultados_seeds.append(
                    {
                        "ventana":
                            nombre_ventana,

                        "train_months":
                            ",".join(
                                str(x)
                                for x
                                in train_months
                            ),

                        "test_month":
                            test_month,

                        "configuracion":
                            config_nombre,

                        "n_features":
                            len(features),

                        "seed":
                            seed,

                        "auc":
                            evaluacion[
                                "auc"
                            ],

                        "average_precision":
                            evaluacion[
                                "average_precision"
                            ],

                        "logloss":
                            evaluacion[
                                "logloss"
                            ],

                        "gain_n12000":
                            evaluacion[
                                "gain_n12000"
                            ],

                        "positivos_n12000":
                            evaluacion[
                                "positivos_n12000"
                            ],

                        "best_n":
                            evaluacion[
                                "best_n"
                            ],

                        "best_gain":
                            evaluacion[
                                "best_gain"
                            ],

                        "elapsed_seconds":
                            elapsed,
                    }
                )

                print(
                    f"[{i}/5] "
                    f"seed={seed} "
                    f"| AUC="
                    f"{evaluacion['auc']:.6f} "
                    f"| AP="
                    f"{evaluacion['average_precision']:.6f} "
                    f"| N12000="
                    f"{evaluacion['gain_n12000']:.3f} M "
                    f"| best="
                    f"{evaluacion['best_n']:,}/"
                    f"{evaluacion['best_gain']:.3f} M "
                    f"| {elapsed:.1f}s",
                    flush=True,
                )

            # =================================================
            # Ensemble
            # =================================================

            matriz = np.column_stack(
                probs_seeds
            )

            probs_ensemble = (
                matriz.mean(
                    axis=1
                )
            )

            ensembles_ventana[
                config_nombre
            ] = probs_ensemble

            evaluacion_ensemble = (
                evaluar(
                    y_true=y_test,
                    probs=probs_ensemble,
                )
            )

            gains_array = np.asarray(
                gains_seeds,
                dtype=float,
            )

            top_ensemble = top_n_ids(
                ids=ids_test,
                probs=probs_ensemble,
            )

            # Ensemble vs cada seed
            jaccards_seed = []

            for probs_seed in probs_seeds:

                top_seed = top_n_ids(
                    ids=ids_test,
                    probs=probs_seed,
                )

                jaccards_seed.append(
                    jaccard(
                        top_ensemble,
                        top_seed,
                    )
                )

            resultado_ensemble = {
                "ventana":
                    nombre_ventana,

                "train_months":
                    ",".join(
                        str(x)
                        for x
                        in train_months
                    ),

                "test_month":
                    test_month,

                "configuracion":
                    config_nombre,

                "n_features":
                    len(features),

                "gain_seed_mean":
                    float(
                        gains_array.mean()
                    ),

                "gain_seed_std":
                    float(
                        gains_array.std(
                            ddof=0
                        )
                    ),

                "gain_seed_min":
                    float(
                        gains_array.min()
                    ),

                "gain_seed_max":
                    float(
                        gains_array.max()
                    ),

                "ensemble_auc":
                    evaluacion_ensemble[
                        "auc"
                    ],

                "ensemble_ap":
                    evaluacion_ensemble[
                        "average_precision"
                    ],

                "ensemble_logloss":
                    evaluacion_ensemble[
                        "logloss"
                    ],

                "ensemble_gain_n12000":
                    evaluacion_ensemble[
                        "gain_n12000"
                    ],

                "ensemble_positivos_n12000":
                    evaluacion_ensemble[
                        "positivos_n12000"
                    ],

                "ensemble_best_n":
                    evaluacion_ensemble[
                        "best_n"
                    ],

                "ensemble_best_gain":
                    evaluacion_ensemble[
                        "best_gain"
                    ],

                "jaccard_ensemble_seed_mean":
                    float(
                        np.mean(
                            jaccards_seed
                        )
                    ),
            }

            resultados_ensembles.append(
                resultado_ensemble
            )

            print(
                "\nResumen:",
                flush=True,
            )

            print(
                f"  Gain seeds : "
                f"{gains_array.mean():.3f} "
                f"± "
                f"{gains_array.std(ddof=0):.3f} M",
                flush=True,
            )

            print(
                f"  Ensemble   : "
                f"{evaluacion_ensemble['gain_n12000']:.3f} M",
                flush=True,
            )

            print(
                f"  AUC ens.   : "
                f"{evaluacion_ensemble['auc']:.6f}",
                flush=True,
            )

            print(
                f"  AP ens.    : "
                f"{evaluacion_ensemble['average_precision']:.6f}",
                flush=True,
            )

            print(
                f"  Best ens.  : "
                f"N="
                f"{evaluacion_ensemble['best_n']:,} "
                f"| "
                f"{evaluacion_ensemble['best_gain']:.3f} M",
                flush=True,
            )

            print(
                f"  Jaccard ens/seeds: "
                f"{np.mean(jaccards_seed):.6f}",
                flush=True,
            )

        # ====================================================
        # Comparación de rankings de configuraciones
        # ====================================================

        print(
            "\n"
            + "-" * 80,
            flush=True,
        )

        print(
            f"JACCARD ENTRE ENSEMBLES - "
            f"{nombre_ventana}",
            flush=True,
        )

        print(
            "-" * 80,
            flush=True,
        )

        nombres = list(
            configuraciones.keys()
        )

        jaccard_configs = []

        for i in range(
            len(nombres)
        ):

            for j in range(
                i + 1,
                len(nombres),
            ):

                nombre_a = nombres[i]
                nombre_b = nombres[j]

                top_a = top_n_ids(
                    ids_test,
                    ensembles_ventana[
                        nombre_a
                    ],
                )

                top_b = top_n_ids(
                    ids_test,
                    ensembles_ventana[
                        nombre_b
                    ],
                )

                jac = jaccard(
                    top_a,
                    top_b,
                )

                interseccion = len(
                    top_a
                    & top_b
                )

                jaccard_configs.append(
                    {
                        "ventana":
                            nombre_ventana,
                        "config_a":
                            nombre_a,
                        "config_b":
                            nombre_b,
                        "jaccard":
                            jac,
                        "interseccion":
                            interseccion,
                    }
                )

                print(
                    f"{nombre_a:<5} vs "
                    f"{nombre_b:<5}: "
                    f"Jaccard={jac:.6f} "
                    f"| intersección="
                    f"{interseccion:,}",
                    flush=True,
                )

        pd.DataFrame(
            jaccard_configs
        ).to_csv(
            OUTPUT_DIR
            / (
                f"jaccard_configuraciones_"
                f"{nombre_ventana}.csv"
            ),
            index=False,
        )

    # ========================================================
    # Guardado general
    # ========================================================

    seeds_df = pd.DataFrame(
        resultados_seeds
    )

    ensembles_df = pd.DataFrame(
        resultados_ensembles
    )

    seeds_df.to_csv(
        OUTPUT_DIR
        / "resultados_seeds.csv",
        index=False,
    )

    ensembles_df.to_csv(
        OUTPUT_DIR
        / "resultados_ensembles.csv",
        index=False,
    )

    # ========================================================
    # Comparación contra BASE
    # ========================================================

    comparaciones = []

    for ventana in (
        ensembles_df[
            "ventana"
        ].unique()
    ):

        sub = (
            ensembles_df.loc[
                ensembles_df[
                    "ventana"
                ].eq(
                    ventana
                )
            ]
            .copy()
        )

        base = (
            sub.loc[
                sub[
                    "configuracion"
                ].eq(
                    "BASE"
                )
            ]
            .iloc[0]
        )

        for _, fila in (
            sub.iterrows()
        ):

            delta_gain = (
                fila[
                    "ensemble_gain_n12000"
                ]
                - base[
                    "ensemble_gain_n12000"
                ]
            )

            delta_pct = (
                100.0
                * delta_gain
                / base[
                    "ensemble_gain_n12000"
                ]
            )

            comparaciones.append(
                {
                    "ventana":
                        ventana,

                    "configuracion":
                        fila[
                            "configuracion"
                        ],

                    "n_features":
                        int(
                            fila[
                                "n_features"
                            ]
                        ),

                    "gain_n12000":
                        fila[
                            "ensemble_gain_n12000"
                        ],

                    "delta_vs_base":
                        delta_gain,

                    "delta_pct_vs_base":
                        delta_pct,

                    "auc":
                        fila[
                            "ensemble_auc"
                        ],

                    "delta_auc_vs_base":
                        (
                            fila[
                                "ensemble_auc"
                            ]
                            - base[
                                "ensemble_auc"
                            ]
                        ),

                    "ap":
                        fila[
                            "ensemble_ap"
                        ],

                    "delta_ap_vs_base":
                        (
                            fila[
                                "ensemble_ap"
                            ]
                            - base[
                                "ensemble_ap"
                            ]
                        ),
                }
            )

    comparacion_df = pd.DataFrame(
        comparaciones
    )

    comparacion_df.to_csv(
        OUTPUT_DIR
        / "comparacion_vs_base.csv",
        index=False,
    )

    # ========================================================
    # Resumen agregado de las dos ventanas
    # ========================================================

    resumen_global = []

    for config_nombre in (
        configuraciones.keys()
    ):

        sub = (
            comparacion_df.loc[
                comparacion_df[
                    "configuracion"
                ].eq(
                    config_nombre
                )
            ]
        )

        resumen_global.append(
            {
                "configuracion":
                    config_nombre,

                "n_features":
                    int(
                        sub[
                            "n_features"
                        ].iloc[0]
                    ),

                "gain_n12000_mean":
                    float(
                        sub[
                            "gain_n12000"
                        ].mean()
                    ),

                "delta_vs_base_mean":
                    float(
                        sub[
                            "delta_vs_base"
                        ].mean()
                    ),

                "delta_pct_vs_base_mean":
                    float(
                        sub[
                            "delta_pct_vs_base"
                        ].mean()
                    ),

                "delta_auc_vs_base_mean":
                    float(
                        sub[
                            "delta_auc_vs_base"
                        ].mean()
                    ),

                "delta_ap_vs_base_mean":
                    float(
                        sub[
                            "delta_ap_vs_base"
                        ].mean()
                    ),

                "wins_vs_base":
                    int(
                        (
                            sub[
                                "delta_vs_base"
                            ]
                            > 0
                        ).sum()
                    ),

                "ties_vs_base":
                    int(
                        np.isclose(
                            sub[
                                "delta_vs_base"
                            ],
                            0.0,
                        ).sum()
                    ),
            }
        )

    resumen_global_df = pd.DataFrame(
        resumen_global
    )

    resumen_global_df.to_csv(
        OUTPUT_DIR
        / "resumen_global.csv",
        index=False,
    )

    # ========================================================
    # Salida final
    # ========================================================

    print(
        "\n\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "COMPARACIÓN ENSEMBLES VS BASE",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    for ventana in (
        comparacion_df[
            "ventana"
        ].unique()
    ):

        print(
            f"\nVENTANA {ventana}",
            flush=True,
        )

        sub = (
            comparacion_df.loc[
                comparacion_df[
                    "ventana"
                ].eq(
                    ventana
                )
            ]
        )

        for _, fila in (
            sub.iterrows()
        ):

            print(
                f"  "
                f"{fila['configuracion']:<5} "
                f"| features="
                f"{int(fila['n_features']):>3} "
                f"| gain="
                f"{fila['gain_n12000']:.3f} M "
                f"| ΔBASE="
                f"{fila['delta_vs_base']:+.3f} M "
                f"("
                f"{fila['delta_pct_vs_base']:+.2f}%"
                f") "
                f"| ΔAUC="
                f"{fila['delta_auc_vs_base']:+.6f} "
                f"| ΔAP="
                f"{fila['delta_ap_vs_base']:+.6f}",
                flush=True,
            )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "RESUMEN GLOBAL z528",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    for _, fila in (
        resumen_global_df.iterrows()
    ):

        print(
            f"{fila['configuracion']:<5} "
            f"| features="
            f"{int(fila['n_features']):>3} "
            f"| gain medio="
            f"{fila['gain_n12000_mean']:.3f} M "
            f"| ΔBASE medio="
            f"{fila['delta_vs_base_mean']:+.3f} M "
            f"| wins="
            f"{int(fila['wins_vs_base'])}/2 "
            f"| ties="
            f"{int(fila['ties_vs_base'])}/2",
            flush=True,
        )

    # ========================================================
    # Lectura automática prudente
    # ========================================================

    no_base = (
        resumen_global_df.loc[
            ~resumen_global_df[
                "configuracion"
            ].eq(
                "BASE"
            )
        ]
        .sort_values(
            "delta_vs_base_mean",
            ascending=False,
        )
    )

    mejor = no_base.iloc[0]

    print(
        "\nMejor configuración histórica "
        "por Δ gain medio vs BASE:",
        flush=True,
    )

    print(
        f"  {mejor['configuracion']} "
        f"| Δ medio="
        f"{mejor['delta_vs_base_mean']:+.3f} M "
        f"| wins="
        f"{int(mejor['wins_vs_base'])}/2",
        flush=True,
    )

    if (
        int(
            mejor[
                "wins_vs_base"
            ]
        )
        == 2
        and mejor[
            "delta_vs_base_mean"
        ]
        > 0
    ):

        print(
            "  Señal consistente en "
            "ambos holdouts.",
            flush=True,
        )

    else:

        print(
            "  La evidencia no es uniforme "
            "en ambos holdouts.",
            flush=True,
        )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "script":
            "z528_competencia_01_ablation_feature_engineering.py",

        "dataset":
            str(
                DATASET
            ),

        "target":
            "BAJA+2",

        "seeds":
            SEEDS,

        "n_fijo":
            N_FIJO,

        "configuraciones": {
            nombre:
                len(features)
            for nombre, features
            in configuraciones.items()
        },

        "ventanas":
            VENTANAS,

        "params":
            PARAMS_BASE,
    }

    with (
        OUTPUT_DIR
        / "metadata_z528.json"
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
        time.time()
        - inicio
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
        "202108 ni el Public LB.",
        flush=True,
    )


if __name__ == "__main__":
    main()