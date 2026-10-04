#!/usr/bin/env python3
"""
z525_competencia_01_robustez_feature_engineering.py

Robustez del Feature Engineering Histórico frente a random_state.

Compara en dos ventanas OOT:

A) BASELINE
   152 variables originales

B) HISTÓRICO
   152 originales
 + 152 lag1
 + 152 delta_lag1
 +   1 lag1_disponible
 = 457 variables

Seeds:
    290497
    100003
    200003
    300007
    400009

Ventanas:
    202104          -> 202105
    202104-202105   -> 202106

Para cada arquitectura y ventana:
    - entrena 5 seeds
    - calcula AUC, AP, LogLoss
    - calcula gain a N=12000
    - calcula mejor gain de la grilla
    - promedia probabilidades de las 5 seeds
    - evalúa el ensemble
    - mide estabilidad entre seeds

NO usa:
    - 202107 como target
    - 202108
    - Public Leaderboard

Objetivo:
determinar si la mejora observada en z524 es robusta a la seed
antes de generar un nuevo candidato de competencia.
"""

from __future__ import annotations

import itertools
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
    "competencia_01/robustez_feature_engineering_z525"
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
        "nombre": "train_202104_test_202105",
        "train_months": [
            202104,
        ],
        "test_month": 202105,
    },
    {
        "nombre": "train_202104_202105_test_202106",
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
        "auc":
            float(
                roc_auc_score(
                    y_true,
                    probs,
                )
            ),
        "average_precision":
            float(
                average_precision_score(
                    y_true,
                    probs,
                )
            ),
        "logloss":
            float(
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
            n
            - positivos
        )

        gain = (
            positivos
            * GANANCIA_ACIERTO
            + negativos
            * COSTO_ERROR
        )

        filas.append(
            {
                "n":
                    n,
                "gain":
                    float(gain),
                "gain_millones":
                    float(
                        gain
                        / 1_000_000
                    ),
                "positivos":
                    positivos,
                "negativos":
                    negativos,
                "precision":
                    (
                        positivos
                        / n
                    ),
                "recall":
                    (
                        positivos
                        / total_positivos
                        if total_positivos > 0
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        filas
    )


def evaluar_probabilidades(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict:

    mets = calcular_metricas(
        y_true=y_true,
        probs=probs,
    )

    curva = curva_ganancia(
        y_true=y_true,
        probs=probs,
    )

    mejor = curva.loc[
        curva[
            "gain"
        ].idxmax()
    ]

    fijo = curva.loc[
        curva[
            "n"
        ].eq(
            N_FIJO
        )
    ]

    if len(fijo) != 1:
        raise ValueError(
            "No se encontró "
            "N=12000 en la curva."
        )

    fijo = fijo.iloc[0]

    return {
        "auc":
            mets["auc"],
        "average_precision":
            mets[
                "average_precision"
            ],
        "logloss":
            mets["logloss"],
        "best_n":
            int(
                mejor["n"]
            ),
        "best_gain_millones":
            float(
                mejor[
                    "gain_millones"
                ]
            ),
        "gain_n12000_millones":
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
    }


# ============================================================
# Ranking / estabilidad
# ============================================================

def top_n_ids(
    ids: np.ndarray,
    probs: np.ndarray,
    n: int = N_FIJO,
) -> set[int]:

    order = np.argsort(
        -probs,
        kind="mergesort",
    )

    return set(
        ids[
            order[:n]
        ].tolist()
    )


def jaccard(
    a: set[int],
    b: set[int],
) -> float:

    union = (
        a | b
    )

    if not union:
        return 1.0

    return (
        len(
            a & b
        )
        / len(
            union
        )
    )


def estabilidad_seeds(
    tops: dict[int, set[int]],
) -> dict:

    valores = []

    detalle = []

    for (
        seed_a,
        seed_b,
    ) in itertools.combinations(
        sorted(
            tops.keys()
        ),
        2,
    ):

        jac = jaccard(
            tops[seed_a],
            tops[seed_b],
        )

        valores.append(
            jac
        )

        detalle.append(
            {
                "seed_a":
                    seed_a,
                "seed_b":
                    seed_b,
                "jaccard":
                    jac,
            }
        )

    valores_np = np.asarray(
        valores,
        dtype=float,
    )

    return {
        "mean":
            float(
                valores_np.mean()
            ),
        "std":
            float(
                valores_np.std(
                    ddof=0
                )
            ),
        "min":
            float(
                valores_np.min()
            ),
        "max":
            float(
                valores_np.max()
            ),
        "detalle":
            detalle,
    }


# ============================================================
# Features
# ============================================================

def identificar_features(
    df: pd.DataFrame,
):

    excluir = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
        LAG_AVAILABLE_COL,
    }

    features_delta = [
        c
        for c in df.columns
        if c.endswith(
            "_delta_lag1"
        )
    ]

    features_lag = [
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

    features_originales = [
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

    features_historicas = (
        features_originales
        + features_lag
        + features_delta
        + [
            LAG_AVAILABLE_COL
        ]
    )

    return (
        features_originales,
        features_lag,
        features_delta,
        features_historicas,
    )


# ============================================================
# Entrenamiento de una arquitectura
# ============================================================

def evaluar_arquitectura(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    ids_test: np.ndarray,
    arquitectura: str,
    ventana_dir: Path,
):

    print(
        "\n"
        + "-" * 80,
        flush=True,
    )

    print(
        f"ARQUITECTURA: "
        f"{arquitectura.upper()}",
        flush=True,
    )

    print(
        "-" * 80,
        flush=True,
    )

    print(
        f"Features: "
        f"{X_train.shape[1]:,}",
        flush=True,
    )

    resultados = []
    probabilidades = {}
    tops = {}

    for i, seed in enumerate(
        SEEDS,
        start=1,
    ):

        print(
            f"\n[{i}/{len(SEEDS)}] "
            f"Seed {seed}",
            flush=True,
        )

        params = (
            PARAMS_BASE.copy()
        )

        params[
            "random_state"
        ] = seed

        modelo = LGBMClassifier(
            **params
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

        evaluacion = (
            evaluar_probabilidades(
                y_true=y_test,
                probs=probs,
            )
        )

        evaluacion.update(
            {
                "arquitectura":
                    arquitectura,
                "seed":
                    seed,
                "n_features":
                    int(
                        X_train.shape[1]
                    ),
                "elapsed_seconds":
                    float(
                        elapsed
                    ),
            }
        )

        resultados.append(
            evaluacion
        )

        probabilidades[
            seed
        ] = probs

        tops[
            seed
        ] = top_n_ids(
            ids=ids_test,
            probs=probs,
        )

        print(
            f"  AUC      : "
            f"{evaluacion['auc']:.6f}",
            flush=True,
        )

        print(
            f"  AP       : "
            f"{evaluacion['average_precision']:.6f}",
            flush=True,
        )

        print(
            f"  N=12.000 : "
            f"{evaluacion['gain_n12000_millones']:.3f} M "
            f"| pos="
            f"{evaluacion['positivos_n12000']}",
            flush=True,
        )

        print(
            f"  Best     : "
            f"N={evaluacion['best_n']:,} "
            f"| "
            f"{evaluacion['best_gain_millones']:.3f} M",
            flush=True,
        )

        print(
            f"  Tiempo   : "
            f"{elapsed:.1f} s",
            flush=True,
        )

    resultados_df = pd.DataFrame(
        resultados
    )

    # --------------------------------------------------------
    # Ensemble de probabilidades
    # --------------------------------------------------------

    matriz_probs = np.column_stack(
        [
            probabilidades[
                seed
            ]
            for seed in SEEDS
        ]
    )

    probs_ensemble = (
        matriz_probs.mean(
            axis=1
        )
    )

    ensemble_eval = (
        evaluar_probabilidades(
            y_true=y_test,
            probs=probs_ensemble,
        )
    )

    ensemble_top = top_n_ids(
        ids=ids_test,
        probs=probs_ensemble,
    )

    # --------------------------------------------------------
    # Estabilidad entre seeds
    # --------------------------------------------------------

    estabilidad = (
        estabilidad_seeds(
            tops
        )
    )

    # Jaccard ensemble vs cada seed
    ensemble_vs_seed = []

    for seed in SEEDS:

        jac = jaccard(
            ensemble_top,
            tops[seed],
        )

        ensemble_vs_seed.append(
            {
                "seed":
                    seed,
                "jaccard_ensemble":
                    jac,
            }
        )

    ensemble_vs_seed_df = (
        pd.DataFrame(
            ensemble_vs_seed
        )
    )

    # --------------------------------------------------------
    # Consenso TOP 12000
    # --------------------------------------------------------

    contador = {}

    for seed in SEEDS:

        for cliente in tops[
            seed
        ]:

            contador[
                cliente
            ] = (
                contador.get(
                    cliente,
                    0,
                )
                + 1
            )

    consenso_ensemble = []

    for cliente in ensemble_top:

        consenso_ensemble.append(
            contador.get(
                cliente,
                0,
            )
        )

    consenso_ensemble = np.asarray(
        consenso_ensemble,
        dtype=int,
    )

    consenso_rows = []

    for k in range(
        1,
        len(SEEDS) + 1,
    ):

        cantidad = int(
            np.sum(
                consenso_ensemble
                == k
            )
        )

        consenso_rows.append(
            {
                "cantidad_seeds":
                    k,
                "clientes_ensemble":
                    cantidad,
                "porcentaje":
                    100.0
                    * cantidad
                    / N_FIJO,
            }
        )

    consenso_df = pd.DataFrame(
        consenso_rows
    )

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    gains = (
        resultados_df[
            "gain_n12000_millones"
        ]
        .to_numpy()
    )

    aucs = (
        resultados_df[
            "auc"
        ]
        .to_numpy()
    )

    aps = (
        resultados_df[
            "average_precision"
        ]
        .to_numpy()
    )

    resumen = {
        "arquitectura":
            arquitectura,
        "n_features":
            int(
                X_train.shape[1]
            ),
        "gain_seed_mean":
            float(
                gains.mean()
            ),
        "gain_seed_std":
            float(
                gains.std(
                    ddof=0
                )
            ),
        "gain_seed_min":
            float(
                gains.min()
            ),
        "gain_seed_max":
            float(
                gains.max()
            ),
        "auc_seed_mean":
            float(
                aucs.mean()
            ),
        "auc_seed_std":
            float(
                aucs.std(
                    ddof=0
                )
            ),
        "ap_seed_mean":
            float(
                aps.mean()
            ),
        "ap_seed_std":
            float(
                aps.std(
                    ddof=0
                )
            ),
        "ensemble_auc":
            ensemble_eval[
                "auc"
            ],
        "ensemble_ap":
            ensemble_eval[
                "average_precision"
            ],
        "ensemble_logloss":
            ensemble_eval[
                "logloss"
            ],
        "ensemble_gain_n12000":
            ensemble_eval[
                "gain_n12000_millones"
            ],
        "ensemble_positivos_n12000":
            ensemble_eval[
                "positivos_n12000"
            ],
        "ensemble_best_n":
            ensemble_eval[
                "best_n"
            ],
        "ensemble_best_gain":
            ensemble_eval[
                "best_gain_millones"
            ],
        "jaccard_seeds_mean":
            estabilidad[
                "mean"
            ],
        "jaccard_seeds_std":
            estabilidad[
                "std"
            ],
        "jaccard_seeds_min":
            estabilidad[
                "min"
            ],
        "jaccard_seeds_max":
            estabilidad[
                "max"
            ],
        "jaccard_ensemble_mean":
            float(
                ensemble_vs_seed_df[
                    "jaccard_ensemble"
                ].mean()
            ),
        "jaccard_ensemble_min":
            float(
                ensemble_vs_seed_df[
                    "jaccard_ensemble"
                ].min()
            ),
        "jaccard_ensemble_max":
            float(
                ensemble_vs_seed_df[
                    "jaccard_ensemble"
                ].max()
            ),
    }

    print(
        "\nResumen seeds:",
        flush=True,
    )

    print(
        f"  Gain N=12000 "
        f"media : "
        f"{resumen['gain_seed_mean']:.3f} M",
        flush=True,
    )

    print(
        f"  Gain N=12000 "
        f"std   : "
        f"{resumen['gain_seed_std']:.3f} M",
        flush=True,
    )

    print(
        f"  Gain N=12000 "
        f"rango : "
        f"{resumen['gain_seed_min']:.3f} - "
        f"{resumen['gain_seed_max']:.3f} M",
        flush=True,
    )

    print(
        f"  Jaccard seeds "
        f"media: "
        f"{resumen['jaccard_seeds_mean']:.6f}",
        flush=True,
    )

    print(
        "\nEnsemble 5 seeds:",
        flush=True,
    )

    print(
        f"  AUC      : "
        f"{ensemble_eval['auc']:.6f}",
        flush=True,
    )

    print(
        f"  AP       : "
        f"{ensemble_eval['average_precision']:.6f}",
        flush=True,
    )

    print(
        f"  N=12.000 : "
        f"{ensemble_eval['gain_n12000_millones']:.3f} M "
        f"| pos="
        f"{ensemble_eval['positivos_n12000']}",
        flush=True,
    )

    print(
        f"  Best     : "
        f"N={ensemble_eval['best_n']:,} "
        f"| "
        f"{ensemble_eval['best_gain_millones']:.3f} M",
        flush=True,
    )

    print(
        f"  Jaccard ensemble vs seeds: "
        f"{resumen['jaccard_ensemble_mean']:.6f}",
        flush=True,
    )

    # --------------------------------------------------------
    # Guardado
    # --------------------------------------------------------

    arquitectura_dir = (
        ventana_dir
        / arquitectura
    )

    arquitectura_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    resultados_df.to_csv(
        arquitectura_dir
        / "resultados_seeds.csv",
        index=False,
    )

    pd.DataFrame(
        estabilidad[
            "detalle"
        ]
    ).to_csv(
        arquitectura_dir
        / "jaccard_entre_seeds.csv",
        index=False,
    )

    ensemble_vs_seed_df.to_csv(
        arquitectura_dir
        / "jaccard_ensemble_vs_seed.csv",
        index=False,
    )

    consenso_df.to_csv(
        arquitectura_dir
        / "consenso_ensemble.csv",
        index=False,
    )

    pd.DataFrame(
        {
            ID_COL:
                ids_test,
            "target":
                y_test,
            "prob_ensemble":
                probs_ensemble,
        }
    ).sort_values(
        [
            "prob_ensemble",
            ID_COL,
        ],
        ascending=[
            False,
            True,
        ],
    ).to_csv(
        arquitectura_dir
        / "ranking_ensemble.csv",
        index=False,
    )

    with (
        arquitectura_dir
        / "resumen.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            resumen,
            f,
            indent=2,
        )

    return (
        resumen,
        resultados_df,
        probs_ensemble,
        ensemble_top,
    )


# ============================================================
# Evaluación de una ventana
# ============================================================

def evaluar_ventana(
    df: pd.DataFrame,
    features_originales: list[str],
    features_historicas: list[str],
    ventana: dict,
):

    nombre = (
        ventana[
            "nombre"
        ]
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
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        f"VENTANA: {nombre}",
        flush=True,
    )

    print(
        "=" * 80,
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
        or test[
            TARGET_COL
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            f"{nombre}: "
            "target NA."
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

    print(
        f"Train lag disponible: "
        f"{train[LAG_AVAILABLE_COL].mean() * 100:.3f}%",
        flush=True,
    )

    print(
        f"Test lag disponible : "
        f"{test[LAG_AVAILABLE_COL].mean() * 100:.3f}%",
        flush=True,
    )

    ventana_dir = (
        OUTPUT_DIR
        / nombre
    )

    ventana_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    (
        resumen_base,
        resultados_base,
        probs_base,
        top_base,
    ) = evaluar_arquitectura(
        X_train=train[
            features_originales
        ],
        y_train=y_train,
        X_test=test[
            features_originales
        ],
        y_test=y_test,
        ids_test=ids_test,
        arquitectura="baseline",
        ventana_dir=ventana_dir,
    )

    # --------------------------------------------------------
    # Histórico
    # --------------------------------------------------------

    (
        resumen_hist,
        resultados_hist,
        probs_hist,
        top_hist,
    ) = evaluar_arquitectura(
        X_train=train[
            features_historicas
        ],
        y_train=y_train,
        X_test=test[
            features_historicas
        ],
        y_test=y_test,
        ids_test=ids_test,
        arquitectura="historico",
        ventana_dir=ventana_dir,
    )

    # --------------------------------------------------------
    # Comparación seed por seed
    # --------------------------------------------------------

    pares = (
        resultados_base[
            [
                "seed",
                "gain_n12000_millones",
                "auc",
                "average_precision",
            ]
        ]
        .merge(
            resultados_hist[
                [
                    "seed",
                    "gain_n12000_millones",
                    "auc",
                    "average_precision",
                ]
            ],
            on="seed",
            suffixes=(
                "_baseline",
                "_historico",
            ),
        )
    )

    pares[
        "delta_gain_millones"
    ] = (
        pares[
            "gain_n12000_millones_historico"
        ]
        - pares[
            "gain_n12000_millones_baseline"
        ]
    )

    pares[
        "delta_gain_pct"
    ] = (
        100.0
        * pares[
            "delta_gain_millones"
        ]
        / pares[
            "gain_n12000_millones_baseline"
        ]
    )

    pares[
        "delta_auc"
    ] = (
        pares[
            "auc_historico"
        ]
        - pares[
            "auc_baseline"
        ]
    )

    pares[
        "delta_ap"
    ] = (
        pares[
            "average_precision_historico"
        ]
        - pares[
            "average_precision_baseline"
        ]
    )

    pares[
        "historico_gana"
    ] = (
        pares[
            "delta_gain_millones"
        ]
        > 0
    )

    pares.to_csv(
        ventana_dir
        / "comparacion_seed_a_seed.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Comparación de ensembles
    # --------------------------------------------------------

    delta_ensemble = (
        resumen_hist[
            "ensemble_gain_n12000"
        ]
        - resumen_base[
            "ensemble_gain_n12000"
        ]
    )

    delta_ensemble_pct = (
        100.0
        * delta_ensemble
        / resumen_base[
            "ensemble_gain_n12000"
        ]
    )

    jaccard_ensembles = jaccard(
        top_base,
        top_hist,
    )

    interseccion = len(
        top_base
        & top_hist
    )

    comparacion = {
        "ventana":
            nombre,
        "test_month":
            test_month,
        "baseline_seed_gain_mean":
            resumen_base[
                "gain_seed_mean"
            ],
        "historico_seed_gain_mean":
            resumen_hist[
                "gain_seed_mean"
            ],
        "delta_seed_gain_mean":
            (
                resumen_hist[
                    "gain_seed_mean"
                ]
                - resumen_base[
                    "gain_seed_mean"
                ]
            ),
        "baseline_seed_gain_std":
            resumen_base[
                "gain_seed_std"
            ],
        "historico_seed_gain_std":
            resumen_hist[
                "gain_seed_std"
            ],
        "historico_gana_seeds":
            int(
                pares[
                    "historico_gana"
                ].sum()
            ),
        "total_seeds":
            len(SEEDS),
        "baseline_ensemble_gain":
            resumen_base[
                "ensemble_gain_n12000"
            ],
        "historico_ensemble_gain":
            resumen_hist[
                "ensemble_gain_n12000"
            ],
        "delta_ensemble_gain":
            delta_ensemble,
        "delta_ensemble_gain_pct":
            delta_ensemble_pct,
        "baseline_ensemble_auc":
            resumen_base[
                "ensemble_auc"
            ],
        "historico_ensemble_auc":
            resumen_hist[
                "ensemble_auc"
            ],
        "baseline_ensemble_ap":
            resumen_base[
                "ensemble_ap"
            ],
        "historico_ensemble_ap":
            resumen_hist[
                "ensemble_ap"
            ],
        "jaccard_ensembles_n12000":
            jaccard_ensembles,
        "interseccion_ensembles_n12000":
            interseccion,
    }

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "COMPARACIÓN ROBUSTA "
        "BASELINE vs HISTÓRICO",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "\nSeed por seed:",
        flush=True,
    )

    print(
        pares[
            [
                "seed",
                "gain_n12000_millones_baseline",
                "gain_n12000_millones_historico",
                "delta_gain_millones",
                "delta_gain_pct",
                "historico_gana",
            ]
        ].to_string(
            index=False,
            formatters={
                "gain_n12000_millones_baseline":
                    lambda x:
                        f"{x:.3f}",
                "gain_n12000_millones_historico":
                    lambda x:
                        f"{x:.3f}",
                "delta_gain_millones":
                    lambda x:
                        f"{x:+.3f}",
                "delta_gain_pct":
                    lambda x:
                        f"{x:+.2f}%",
            },
        ),
        flush=True,
    )

    print(
        f"\nHistórico gana seeds: "
        f"{comparacion['historico_gana_seeds']}"
        f"/{len(SEEDS)}",
        flush=True,
    )

    print(
        f"Media baseline : "
        f"{comparacion['baseline_seed_gain_mean']:.3f} M "
        f"± "
        f"{comparacion['baseline_seed_gain_std']:.3f}",
        flush=True,
    )

    print(
        f"Media histórico: "
        f"{comparacion['historico_seed_gain_mean']:.3f} M "
        f"± "
        f"{comparacion['historico_seed_gain_std']:.3f}",
        flush=True,
    )

    print(
        f"Δ media        : "
        f"{comparacion['delta_seed_gain_mean']:+.3f} M",
        flush=True,
    )

    print(
        "\nEnsemble de probabilidades:",
        flush=True,
    )

    print(
        f"  Baseline : "
        f"{comparacion['baseline_ensemble_gain']:.3f} M",
        flush=True,
    )

    print(
        f"  Histórico: "
        f"{comparacion['historico_ensemble_gain']:.3f} M",
        flush=True,
    )

    print(
        f"  Δ        : "
        f"{comparacion['delta_ensemble_gain']:+.3f} M "
        f"("
        f"{comparacion['delta_ensemble_gain_pct']:+.2f}%"
        f")",
        flush=True,
    )

    print(
        f"  Jaccard ensembles: "
        f"{jaccard_ensembles:.6f}",
        flush=True,
    )

    with (
        ventana_dir
        / "comparacion_final.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            comparacion,
            f,
            indent=2,
        )

    return comparacion


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
        "z525 - ROBUSTEZ FEATURE ENGINEERING HISTÓRICO",
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
        f"Corte fijo: "
        f"N={N_FIJO:,}",
        flush=True,
    )

    print(
        "\nLeyendo Parquet...",
        flush=True,
    )

    df = pd.read_parquet(
        DATASET
    )

    # Solamente los meses de validación.
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
        features_historicas,
    ) = identificar_features(
        df
    )

    print(
        f"\nFilas cargadas       : "
        f"{len(df):,}",
        flush=True,
    )

    print(
        f"Features originales  : "
        f"{len(features_originales):,}",
        flush=True,
    )

    print(
        f"Features lag1        : "
        f"{len(features_lag):,}",
        flush=True,
    )

    print(
        f"Features delta_lag1  : "
        f"{len(features_delta):,}",
        flush=True,
    )

    print(
        f"Features históricas  : "
        f"{len(features_historicas):,}",
        flush=True,
    )

    # --------------------------------------------------------
    # Auditorías
    # --------------------------------------------------------

    if len(
        features_originales
    ) != 152:
        raise ValueError(
            "Cantidad incorrecta "
            "de features originales."
        )

    if len(
        features_lag
    ) != 152:
        raise ValueError(
            "Cantidad incorrecta "
            "de features lag1."
        )

    if len(
        features_delta
    ) != 152:
        raise ValueError(
            "Cantidad incorrecta "
            "de features delta."
        )

    if len(
        features_historicas
    ) != 457:
        raise ValueError(
            "Cantidad incorrecta "
            "de features históricas."
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
            "Hay duplicados "
            "cliente/mes."
        )

    # --------------------------------------------------------
    # Experimentos
    # --------------------------------------------------------

    comparaciones = []

    for ventana in VENTANAS:

        comparacion = (
            evaluar_ventana(
                df=df,
                features_originales=
                    features_originales,
                features_historicas=
                    features_historicas,
                ventana=ventana,
            )
        )

        comparaciones.append(
            comparacion
        )

    # --------------------------------------------------------
    # Resumen final
    # --------------------------------------------------------

    resumen = pd.DataFrame(
        comparaciones
    )

    resumen.to_csv(
        OUTPUT_DIR
        / "resumen_z525.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "RESUMEN FINAL z525",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    columnas = [
        "test_month",
        "historico_gana_seeds",
        "total_seeds",
        "baseline_seed_gain_mean",
        "historico_seed_gain_mean",
        "delta_seed_gain_mean",
        "baseline_seed_gain_std",
        "historico_seed_gain_std",
        "baseline_ensemble_gain",
        "historico_ensemble_gain",
        "delta_ensemble_gain",
        "delta_ensemble_gain_pct",
        "jaccard_ensembles_n12000",
    ]

    print(
        "\n"
        + resumen[
            columnas
        ].to_string(
            index=False,
            formatters={
                "baseline_seed_gain_mean":
                    lambda x:
                        f"{x:.3f}",
                "historico_seed_gain_mean":
                    lambda x:
                        f"{x:.3f}",
                "delta_seed_gain_mean":
                    lambda x:
                        f"{x:+.3f}",
                "baseline_seed_gain_std":
                    lambda x:
                        f"{x:.3f}",
                "historico_seed_gain_std":
                    lambda x:
                        f"{x:.3f}",
                "baseline_ensemble_gain":
                    lambda x:
                        f"{x:.3f}",
                "historico_ensemble_gain":
                    lambda x:
                        f"{x:.3f}",
                "delta_ensemble_gain":
                    lambda x:
                        f"{x:+.3f}",
                "delta_ensemble_gain_pct":
                    lambda x:
                        f"{x:+.2f}%",
                "jaccard_ensembles_n12000":
                    lambda x:
                        f"{x:.6f}",
            },
        ),
        flush=True,
    )

    total_ganadas = int(
        resumen[
            "historico_gana_seeds"
        ].sum()
    )

    total_comparaciones = (
        len(SEEDS)
        * len(VENTANAS)
    )

    ensembles_ganados = int(
        (
            resumen[
                "delta_ensemble_gain"
            ]
            > 0
        ).sum()
    )

    print(
        "\nDiagnóstico global:",
        flush=True,
    )

    print(
        f"  Histórico gana seed-a-seed: "
        f"{total_ganadas}/"
        f"{total_comparaciones}",
        flush=True,
    )

    print(
        f"  Histórico gana ensembles   : "
        f"{ensembles_ganados}/"
        f"{len(VENTANAS)}",
        flush=True,
    )

    print(
        f"  Δ medio ensembles          : "
        f"{resumen['delta_ensemble_gain'].mean():+.3f} M",
        flush=True,
    )

    print(
        "\nCriterio de lectura:",
        flush=True,
    )

    print(
        "  Buscamos consistencia entre "
        "ventanas y seeds, no maximizar "
        "una observación aislada.",
        flush=True,
    )

    print(
        "  Si el histórico mantiene la "
        "ventaja en ambas ventanas, "
        "recién entonces corresponde "
        "entrenarlo para 202108.",
        flush=True,
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
        f"\nResultados:\n"
        f"  {OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()