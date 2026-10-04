#!/usr/bin/env python3
"""
z526_competencia_01_historico_incluir_primer_mes.py

Experimento controlado para decidir si conviene incluir 202103
en el entrenamiento del modelo con Feature Engineering histórico.

El dataset histórico tiene:
    152 variables originales
    152 lag1
    152 delta_lag1
      1 lag1_disponible
    -------------------
    457 features

En 202103 no existe mes anterior dentro del dataset:
    - lag1 = NaN
    - delta_lag1 = NaN
    - lag1_disponible = 0

LightGBM puede manejar esos valores faltantes.

Comparación:

A) SIN PRIMER MES
   train: 202104 + 202105
   test : 202106

B) CON PRIMER MES
   train: 202103 + 202104 + 202105
   test : 202106

Todo lo demás permanece idéntico:
    - target BAJA+2
    - 457 features
    - mismos hiperparámetros
    - mismas 5 seeds
    - N fijo = 12000
    - misma grilla de ganancias

Además se comparan los ensembles de probabilidades.

NO usa:
    - 202107 como target
    - 202108
    - Public Leaderboard

Objetivo:
definir si el entrenamiento final histórico debe comenzar
en 202103 o en 202104.
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
    "competencia_01/historico_primer_mes_z526"
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


ESCENARIOS = [
    {
        "nombre": "sin_202103",
        "train_months": [
            202104,
            202105,
        ],
    },
    {
        "nombre": "con_202103",
        "train_months": [
            202103,
            202104,
            202105,
        ],
    },
]

TEST_MONTH = 202106


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
                "gain": float(gain),
                "gain_millones": float(
                    gain / 1_000_000
                ),
                "positivos": positivos,
                "negativos": negativos,
                "precision": (
                    positivos / n
                ),
                "recall": (
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
            "No se encontró N=12000."
        )

    fijo = fijo.iloc[0]

    return {
        "auc": mets["auc"],
        "average_precision":
            mets["average_precision"],
        "logloss": mets["logloss"],
        "best_n": int(
            mejor["n"]
        ),
        "best_gain_millones": float(
            mejor[
                "gain_millones"
            ]
        ),
        "gain_n12000_millones": float(
            fijo[
                "gain_millones"
            ]
        ),
        "positivos_n12000": int(
            fijo[
                "positivos"
            ]
        ),
        "precision_n12000": float(
            fijo[
                "precision"
            ]
        ),
        "recall_n12000": float(
            fijo[
                "recall"
            ]
        ),
    }


# ============================================================
# Ranking
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

    union = a | b

    if not union:
        return 1.0

    return (
        len(a & b)
        / len(union)
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
                "seed_a": seed_a,
                "seed_b": seed_b,
                "jaccard": jac,
            }
        )

    valores = np.asarray(
        valores,
        dtype=float,
    )

    return {
        "mean": float(
            valores.mean()
        ),
        "std": float(
            valores.std(
                ddof=0
            )
        ),
        "min": float(
            valores.min()
        ),
        "max": float(
            valores.max()
        ),
        "detalle": detalle,
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
# Entrenamiento de escenario
# ============================================================

def evaluar_escenario(
    df: pd.DataFrame,
    features: list[str],
    escenario: dict,
    test: pd.DataFrame,
    y_test: np.ndarray,
    ids_test: np.ndarray,
):

    nombre = escenario[
        "nombre"
    ]

    train_months = escenario[
        "train_months"
    ]

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        f"ESCENARIO: {nombre}",
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
        f"Test : {TEST_MONTH}",
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

    if (
        train[
            TARGET_COL
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            f"{nombre}: target NA "
            "en train."
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
        f"Features     : "
        f"{len(features):,}",
        flush=True,
    )

    print(
        "\nCobertura lag por mes "
        "en train:",
        flush=True,
    )

    cobertura = (
        train.groupby(
            MONTH_COL
        )[
            LAG_AVAILABLE_COL
        ]
        .agg(
            [
                "count",
                "sum",
                "mean",
            ]
        )
    )

    cobertura[
        "pct"
    ] = (
        cobertura["mean"]
        * 100.0
    )

    print(
        cobertura[
            [
                "count",
                "sum",
                "pct",
            ]
        ].to_string(),
        flush=True,
    )

    resultados = []
    probabilidades = {}
    tops = {}

    escenario_dir = (
        OUTPUT_DIR
        / nombre
    )

    escenario_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train = train[
        features
    ]

    X_test = test[
        features
    ]

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
                "escenario": nombre,
                "seed": seed,
                "train_rows": int(
                    len(train)
                ),
                "train_positivos": int(
                    y_train.sum()
                ),
                "n_features": int(
                    len(features)
                ),
                "elapsed_seconds": float(
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
            f"  LogLoss  : "
            f"{evaluacion['logloss']:.6f}",
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

    # ========================================================
    # Ensemble
    # ========================================================

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

    # ========================================================
    # Estabilidad
    # ========================================================

    estabilidad = (
        estabilidad_seeds(
            tops
        )
    )

    ensemble_jaccards = []

    for seed in SEEDS:

        ensemble_jaccards.append(
            jaccard(
                ensemble_top,
                tops[seed],
            )
        )

    # ========================================================
    # Resumen
    # ========================================================

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
        "escenario": nombre,
        "train_months": train_months,
        "train_rows": int(
            len(train)
        ),
        "train_positivos": int(
            y_train.sum()
        ),
        "n_features": int(
            len(features)
        ),
        "gain_seed_mean": float(
            gains.mean()
        ),
        "gain_seed_std": float(
            gains.std(
                ddof=0
            )
        ),
        "gain_seed_min": float(
            gains.min()
        ),
        "gain_seed_max": float(
            gains.max()
        ),
        "auc_seed_mean": float(
            aucs.mean()
        ),
        "auc_seed_std": float(
            aucs.std(
                ddof=0
            )
        ),
        "ap_seed_mean": float(
            aps.mean()
        ),
        "ap_seed_std": float(
            aps.std(
                ddof=0
            )
        ),
        "ensemble_auc":
            ensemble_eval["auc"],
        "ensemble_ap":
            ensemble_eval[
                "average_precision"
            ],
        "ensemble_logloss":
            ensemble_eval["logloss"],
        "ensemble_gain_n12000":
            ensemble_eval[
                "gain_n12000_millones"
            ],
        "ensemble_positivos_n12000":
            ensemble_eval[
                "positivos_n12000"
            ],
        "ensemble_best_n":
            ensemble_eval["best_n"],
        "ensemble_best_gain":
            ensemble_eval[
                "best_gain_millones"
            ],
        "jaccard_seeds_mean":
            estabilidad["mean"],
        "jaccard_seeds_std":
            estabilidad["std"],
        "jaccard_ensemble_mean":
            float(
                np.mean(
                    ensemble_jaccards
                )
            ),
    }

    print(
        "\nResumen seeds:",
        flush=True,
    )

    print(
        f"  Gain N=12000 media : "
        f"{resumen['gain_seed_mean']:.3f} M",
        flush=True,
    )

    print(
        f"  Gain N=12000 std   : "
        f"{resumen['gain_seed_std']:.3f} M",
        flush=True,
    )

    print(
        f"  Gain N=12000 rango : "
        f"{resumen['gain_seed_min']:.3f} - "
        f"{resumen['gain_seed_max']:.3f} M",
        flush=True,
    )

    print(
        f"  AUC media          : "
        f"{resumen['auc_seed_mean']:.6f}",
        flush=True,
    )

    print(
        f"  AP media           : "
        f"{resumen['ap_seed_mean']:.6f}",
        flush=True,
    )

    print(
        f"  Jaccard seeds      : "
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
        f"  LogLoss  : "
        f"{ensemble_eval['logloss']:.6f}",
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
        f"  Jaccard ensemble/seeds: "
        f"{resumen['jaccard_ensemble_mean']:.6f}",
        flush=True,
    )

    # ========================================================
    # Guardado
    # ========================================================

    resultados_df.to_csv(
        escenario_dir
        / "resultados_seeds.csv",
        index=False,
    )

    pd.DataFrame(
        estabilidad[
            "detalle"
        ]
    ).to_csv(
        escenario_dir
        / "jaccard_entre_seeds.csv",
        index=False,
    )

    ranking = pd.DataFrame(
        {
            ID_COL: ids_test,
            "target": y_test,
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
    )

    ranking.to_csv(
        escenario_dir
        / "ranking_ensemble.csv",
        index=False,
    )

    with (
        escenario_dir
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

    return {
        "resumen": resumen,
        "resultados": resultados_df,
        "probs_ensemble":
            probs_ensemble,
        "top_ensemble":
            ensemble_top,
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

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "z526 - ¿INCLUIR 202103 EN EL "
        "MODELO HISTÓRICO?",
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
        f"Test fijo: {TEST_MONTH}",
        flush=True,
    )

    print(
        f"Corte fijo: N={N_FIJO:,}",
        flush=True,
    )

    print(
        "\nLeyendo Parquet...",
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
                    202103,
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
            "Se esperaban "
            "152 features originales."
        )

    if len(
        features_lag
    ) != 152:
        raise ValueError(
            "Se esperaban "
            "152 features lag1."
        )

    if len(
        features_delta
    ) != 152:
        raise ValueError(
            "Se esperaban "
            "152 features delta."
        )

    if len(
        features_historicas
    ) != 457:
        raise ValueError(
            "Se esperaban "
            "457 features históricas."
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

    cobertura = (
        df.groupby(
            MONTH_COL
        )[
            LAG_AVAILABLE_COL
        ]
        .agg(
            [
                "count",
                "sum",
                "mean",
            ]
        )
    )

    cobertura[
        "pct"
    ] = (
        cobertura["mean"]
        * 100.0
    )

    print(
        "\nCobertura lag:",
        flush=True,
    )

    print(
        cobertura[
            [
                "count",
                "sum",
                "pct",
            ]
        ].to_string(),
        flush=True,
    )

    # --------------------------------------------------------
    # Test común
    # --------------------------------------------------------

    test = (
        df.loc[
            df[
                MONTH_COL
            ].eq(
                TEST_MONTH
            )
        ]
        .copy()
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

    # --------------------------------------------------------
    # Escenarios
    # --------------------------------------------------------

    resultados = {}

    for escenario in ESCENARIOS:

        resultado = evaluar_escenario(
            df=df,
            features=features_historicas,
            escenario=escenario,
            test=test,
            y_test=y_test,
            ids_test=ids_test,
        )

        resultados[
            escenario[
                "nombre"
            ]
        ] = resultado

    sin_202103 = (
        resultados[
            "sin_202103"
        ]
    )

    con_202103 = (
        resultados[
            "con_202103"
        ]
    )

    # --------------------------------------------------------
    # Comparación seed por seed
    # --------------------------------------------------------

    df_sin = (
        sin_202103[
            "resultados"
        ][
            [
                "seed",
                "gain_n12000_millones",
                "auc",
                "average_precision",
            ]
        ]
    )

    df_con = (
        con_202103[
            "resultados"
        ][
            [
                "seed",
                "gain_n12000_millones",
                "auc",
                "average_precision",
            ]
        ]
    )

    pares = df_sin.merge(
        df_con,
        on="seed",
        suffixes=(
            "_sin202103",
            "_con202103",
        ),
    )

    pares[
        "delta_gain_millones"
    ] = (
        pares[
            "gain_n12000_millones_con202103"
        ]
        - pares[
            "gain_n12000_millones_sin202103"
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
            "gain_n12000_millones_sin202103"
        ]
    )

    pares[
        "delta_auc"
    ] = (
        pares[
            "auc_con202103"
        ]
        - pares[
            "auc_sin202103"
        ]
    )

    pares[
        "delta_ap"
    ] = (
        pares[
            "average_precision_con202103"
        ]
        - pares[
            "average_precision_sin202103"
        ]
    )

    pares[
        "con202103_gana"
    ] = (
        pares[
            "delta_gain_millones"
        ]
        > 0
    )

    pares.to_csv(
        OUTPUT_DIR
        / "comparacion_seed_a_seed.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Comparación ensembles
    # --------------------------------------------------------

    resumen_sin = (
        sin_202103[
            "resumen"
        ]
    )

    resumen_con = (
        con_202103[
            "resumen"
        ]
    )

    delta_ensemble = (
        resumen_con[
            "ensemble_gain_n12000"
        ]
        - resumen_sin[
            "ensemble_gain_n12000"
        ]
    )

    delta_ensemble_pct = (
        100.0
        * delta_ensemble
        / resumen_sin[
            "ensemble_gain_n12000"
        ]
    )

    jaccard_ensembles = jaccard(
        sin_202103[
            "top_ensemble"
        ],
        con_202103[
            "top_ensemble"
        ],
    )

    interseccion = len(
        sin_202103[
            "top_ensemble"
        ]
        & con_202103[
            "top_ensemble"
        ]
    )

    comparacion_final = {
        "test_month": TEST_MONTH,
        "n_fijo": N_FIJO,

        "sin202103_gain_seed_mean":
            resumen_sin[
                "gain_seed_mean"
            ],

        "con202103_gain_seed_mean":
            resumen_con[
                "gain_seed_mean"
            ],

        "delta_gain_seed_mean":
            (
                resumen_con[
                    "gain_seed_mean"
                ]
                - resumen_sin[
                    "gain_seed_mean"
                ]
            ),

        "sin202103_gain_seed_std":
            resumen_sin[
                "gain_seed_std"
            ],

        "con202103_gain_seed_std":
            resumen_con[
                "gain_seed_std"
            ],

        "con202103_gana_seeds":
            int(
                pares[
                    "con202103_gana"
                ].sum()
            ),

        "total_seeds":
            len(SEEDS),

        "sin202103_ensemble_gain":
            resumen_sin[
                "ensemble_gain_n12000"
            ],

        "con202103_ensemble_gain":
            resumen_con[
                "ensemble_gain_n12000"
            ],

        "delta_ensemble_gain":
            delta_ensemble,

        "delta_ensemble_gain_pct":
            delta_ensemble_pct,

        "sin202103_ensemble_auc":
            resumen_sin[
                "ensemble_auc"
            ],

        "con202103_ensemble_auc":
            resumen_con[
                "ensemble_auc"
            ],

        "sin202103_ensemble_ap":
            resumen_sin[
                "ensemble_ap"
            ],

        "con202103_ensemble_ap":
            resumen_con[
                "ensemble_ap"
            ],

        "jaccard_ensembles":
            jaccard_ensembles,

        "interseccion_top12000":
            interseccion,
    }

    with (
        OUTPUT_DIR
        / "comparacion_final.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            comparacion_final,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Salida final
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "COMPARACIÓN FINAL z526",
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
                "gain_n12000_millones_sin202103",
                "gain_n12000_millones_con202103",
                "delta_gain_millones",
                "delta_gain_pct",
                "con202103_gana",
            ]
        ].to_string(
            index=False,
            formatters={
                "gain_n12000_millones_sin202103":
                    lambda x:
                        f"{x:.3f}",
                "gain_n12000_millones_con202103":
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
        f"\nCon 202103 gana seeds: "
        f"{comparacion_final['con202103_gana_seeds']}"
        f"/{len(SEEDS)}",
        flush=True,
    )

    print(
        "\nPromedio de seeds:",
        flush=True,
    )

    print(
        f"  Sin 202103 : "
        f"{comparacion_final['sin202103_gain_seed_mean']:.3f} M "
        f"± "
        f"{comparacion_final['sin202103_gain_seed_std']:.3f}",
        flush=True,
    )

    print(
        f"  Con 202103 : "
        f"{comparacion_final['con202103_gain_seed_mean']:.3f} M "
        f"± "
        f"{comparacion_final['con202103_gain_seed_std']:.3f}",
        flush=True,
    )

    print(
        f"  Δ media    : "
        f"{comparacion_final['delta_gain_seed_mean']:+.3f} M",
        flush=True,
    )

    print(
        "\nEnsemble de probabilidades:",
        flush=True,
    )

    print(
        f"  Sin 202103 : "
        f"{comparacion_final['sin202103_ensemble_gain']:.3f} M",
        flush=True,
    )

    print(
        f"  Con 202103 : "
        f"{comparacion_final['con202103_ensemble_gain']:.3f} M",
        flush=True,
    )

    print(
        f"  Δ          : "
        f"{comparacion_final['delta_ensemble_gain']:+.3f} M "
        f"("
        f"{comparacion_final['delta_ensemble_gain_pct']:+.2f}%"
        f")",
        flush=True,
    )

    print(
        "\nRanking:",
        flush=True,
    )

    print(
        f"  Jaccard ensembles : "
        f"{jaccard_ensembles:.6f}",
        flush=True,
    )

    print(
        f"  Intersección      : "
        f"{interseccion:,}/"
        f"{N_FIJO:,}",
        flush=True,
    )

    print(
        "\nMétricas ensemble:",
        flush=True,
    )

    print(
        f"  AUC sin/con : "
        f"{resumen_sin['ensemble_auc']:.6f} / "
        f"{resumen_con['ensemble_auc']:.6f}",
        flush=True,
    )

    print(
        f"  AP sin/con  : "
        f"{resumen_sin['ensemble_ap']:.6f} / "
        f"{resumen_con['ensemble_ap']:.6f}",
        flush=True,
    )

    print(
        "\nDecisión sugerida:",
        flush=True,
    )

    if (
        comparacion_final[
            "delta_ensemble_gain"
        ]
        > 0
        and comparacion_final[
            "con202103_gana_seeds"
        ]
        >= 3
    ):

        print(
            "  La evidencia favorece incluir "
            "202103 en el entrenamiento final.",
            flush=True,
        )

    elif (
        comparacion_final[
            "delta_ensemble_gain"
        ]
        < 0
        and comparacion_final[
            "con202103_gana_seeds"
        ]
        <= 2
    ):

        print(
            "  La evidencia favorece comenzar "
            "el entrenamiento final en 202104.",
            flush=True,
        )

    else:

        print(
            "  Resultado mixto: revisar gain, "
            "dispersión y estabilidad antes "
            "de decidir.",
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
        f"\nResultados:\n  "
        f"{OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()