#!/usr/bin/env python3
"""
z524_competencia_01_validacion_feature_engineering.py

Validación temporal controlada del Feature Engineering Histórico
construido en z523.

Compara:

A) BASELINE
   152 variables originales

B) HISTÓRICO
   152 variables originales
 + 152 lag1
 + 152 delta_lag1
 +   1 lag1_disponible
 = 457 variables

Todo lo demás permanece constante:
    - target BAJA+2
    - LightGBM baseline
    - seed 290497
    - meses de entrenamiento
    - meses de test
    - grilla de cortes
    - función de ganancia

Ventanas:
    202104          -> 202105
    202104-202105   -> 202106

202103 se excluye deliberadamente porque no dispone de 202102
para construir lag1.

Este script NO utiliza:
    - 202107
    - 202108
    - Public Leaderboard

Objetivo:
aislar el efecto causal experimental de agregar lag1 y delta_lag1
al conjunto de variables.
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
    "competencia_01/validacion_feature_engineering_z524"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
LAG_AVAILABLE_COL = "lag1_disponible"

SEED = 290497

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


BASE_PARAMS = {
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
# Ganancia
# ============================================================

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
                    positivos / n,
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


def metricas(
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


def fila_n(
    curva: pd.DataFrame,
    n: int,
) -> pd.Series:

    resultado = curva.loc[
        curva["n"].eq(n)
    ]

    if len(resultado) != 1:
        raise ValueError(
            f"No se encontró "
            f"exactamente una fila "
            f"para N={n}"
        )

    return resultado.iloc[0]


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
# Identificación de features
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

    lag_suffix = "_lag1"
    delta_suffix = "_delta_lag1"

    features_delta = [
        c
        for c in df.columns
        if c.endswith(
            delta_suffix
        )
    ]

    features_lag = [
        c
        for c in df.columns
        if (
            c.endswith(
                lag_suffix
            )
            and not c.endswith(
                delta_suffix
            )
        )
    ]

    features_originales = [
        c
        for c in df.columns
        if (
            c not in excluir
            and not c.endswith(
                lag_suffix
            )
            and not c.endswith(
                delta_suffix
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
# Entrenamiento
# ============================================================

def entrenar_evaluar(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    ids_test: np.ndarray,
    nombre_modelo: str,
):

    print(
        f"\nEntrenando "
        f"{nombre_modelo}...",
        flush=True,
    )

    print(
        f"  Features: "
        f"{X_train.shape[1]:,}",
        flush=True,
    )

    t0 = time.time()

    modelo = LGBMClassifier(
        **BASE_PARAMS
    )

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

    mets = metricas(
        y_true=y_test,
        probs=probs,
    )

    curva = curva_ganancia(
        y_true=y_test,
        probs=probs,
    )

    mejor = curva.loc[
        curva[
            "gain"
        ].idxmax()
    ]

    fijo = fila_n(
        curva=curva,
        n=N_FIJO,
    )

    order = np.argsort(
        -probs,
        kind="mergesort",
    )

    top_n = set(
        ids_test[
            order[:N_FIJO]
        ].tolist()
    )

    ranking = pd.DataFrame(
        {
            ID_COL:
                ids_test,
            "target":
                y_test,
            "prob":
                probs,
        }
    )

    ranking = (
        ranking
        .sort_values(
            [
                "prob",
                ID_COL,
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
        "rank"
    ] = np.arange(
        1,
        len(ranking) + 1,
        dtype=np.int64,
    )

    resultado = {
        "modelo":
            nombre_modelo,
        "n_features":
            int(
                X_train.shape[1]
            ),
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
        "best_positivos":
            int(
                mejor[
                    "positivos"
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
        "elapsed_seconds":
            float(
                elapsed
            ),
    }

    print(
        f"  AUC      : "
        f"{mets['auc']:.6f}",
        flush=True,
    )

    print(
        f"  AP       : "
        f"{mets['average_precision']:.6f}",
        flush=True,
    )

    print(
        f"  LogLoss  : "
        f"{mets['logloss']:.6f}",
        flush=True,
    )

    print(
        f"  Best     : "
        f"N={int(mejor['n']):,} | "
        f"{mejor['gain_millones']:.3f} M",
        flush=True,
    )

    print(
        f"  N=12.000 : "
        f"{fijo['gain_millones']:.3f} M | "
        f"positivos="
        f"{int(fijo['positivos'])}",
        flush=True,
    )

    print(
        f"  Tiempo   : "
        f"{elapsed:.1f} s",
        flush=True,
    )

    return (
        resultado,
        curva,
        ranking,
        top_n,
    )


# ============================================================
# Ventana experimental
# ============================================================

def evaluar_ventana(
    df: pd.DataFrame,
    features_originales: list[str],
    features_historicas: list[str],
    nombre: str,
    train_months: list[int],
    test_month: int,
):

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

    train[
        "target_binario"
    ] = (
        train[
            TARGET_COL
        ]
        .eq(
            "BAJA+2"
        )
        .astype(
            np.int8
        )
    )

    test[
        "target_binario"
    ] = (
        test[
            TARGET_COL
        ]
        .eq(
            "BAJA+2"
        )
        .astype(
            np.int8
        )
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
            f"{nombre}: hay target NA"
        )

    print(
        f"\nTrain rows   : "
        f"{len(train):,}",
        flush=True,
    )

    print(
        f"Train BAJA+2 : "
        f"{int(train['target_binario'].sum()):,}",
        flush=True,
    )

    print(
        f"Test rows    : "
        f"{len(test):,}",
        flush=True,
    )

    print(
        f"Test BAJA+2  : "
        f"{int(test['target_binario'].sum()):,}",
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

    y_train = (
        train[
            "target_binario"
        ]
        .to_numpy()
    )

    y_test = (
        test[
            "target_binario"
        ]
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

    if len(
        np.unique(
            ids_test
        )
    ) != len(ids_test):
        raise ValueError(
            f"{nombre}: IDs "
            "duplicados en test"
        )

    # --------------------------------------------------------
    # A) Baseline
    # --------------------------------------------------------

    (
        resultado_base,
        curva_base,
        ranking_base,
        top_base,
    ) = entrenar_evaluar(
        X_train=train[
            features_originales
        ],
        y_train=y_train,
        X_test=test[
            features_originales
        ],
        y_test=y_test,
        ids_test=ids_test,
        nombre_modelo="baseline",
    )

    # --------------------------------------------------------
    # B) Histórico
    # --------------------------------------------------------

    (
        resultado_hist,
        curva_hist,
        ranking_hist,
        top_hist,
    ) = entrenar_evaluar(
        X_train=train[
            features_historicas
        ],
        y_train=y_train,
        X_test=test[
            features_historicas
        ],
        y_test=y_test,
        ids_test=ids_test,
        nombre_modelo="historico",
    )

    # --------------------------------------------------------
    # Comparación
    # --------------------------------------------------------

    delta_gain_12000 = (
        resultado_hist[
            "gain_n12000_millones"
        ]
        - resultado_base[
            "gain_n12000_millones"
        ]
    )

    base_gain = (
        resultado_base[
            "gain_n12000_millones"
        ]
    )

    delta_gain_12000_pct = (
        100.0
        * delta_gain_12000
        / base_gain
        if base_gain != 0
        else np.nan
    )

    delta_best_gain = (
        resultado_hist[
            "best_gain_millones"
        ]
        - resultado_base[
            "best_gain_millones"
        ]
    )

    delta_auc = (
        resultado_hist[
            "auc"
        ]
        - resultado_base[
            "auc"
        ]
    )

    delta_ap = (
        resultado_hist[
            "average_precision"
        ]
        - resultado_base[
            "average_precision"
        ]
    )

    jac = jaccard(
        top_base,
        top_hist,
    )

    interseccion = len(
        top_base
        & top_hist
    )

    solo_base = len(
        top_base
        - top_hist
    )

    solo_hist = len(
        top_hist
        - top_base
    )

    print(
        "\n"
        + "-" * 80,
        flush=True,
    )

    print(
        "COMPARACIÓN A/B",
        flush=True,
    )

    print(
        "-" * 80,
        flush=True,
    )

    print(
        f"Δ AUC histórico-baseline       : "
        f"{delta_auc:+.6f}",
        flush=True,
    )

    print(
        f"Δ AP histórico-baseline        : "
        f"{delta_ap:+.6f}",
        flush=True,
    )

    print(
        f"Δ Gain N=12000                 : "
        f"{delta_gain_12000:+.3f} M "
        f"({delta_gain_12000_pct:+.2f}%)",
        flush=True,
    )

    print(
        f"Δ mejor gain                   : "
        f"{delta_best_gain:+.3f} M",
        flush=True,
    )

    print(
        f"Jaccard TOP-12000              : "
        f"{jac:.6f}",
        flush=True,
    )

    print(
        f"Intersección TOP-12000         : "
        f"{interseccion:,}",
        flush=True,
    )

    print(
        f"Solo baseline / solo histórico : "
        f"{solo_base:,} / "
        f"{solo_hist:,}",
        flush=True,
    )

    # --------------------------------------------------------
    # Guardado
    # --------------------------------------------------------

    ventana_dir = (
        OUTPUT_DIR
        / nombre
    )

    ventana_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    resultados_df = pd.DataFrame(
        [
            resultado_base,
            resultado_hist,
        ]
    )

    resultados_df.to_csv(
        ventana_dir
        / "resultados_modelos.csv",
        index=False,
    )

    curva_base.assign(
        modelo="baseline"
    ).to_csv(
        ventana_dir
        / "curva_ganancia_baseline.csv",
        index=False,
    )

    curva_hist.assign(
        modelo="historico"
    ).to_csv(
        ventana_dir
        / "curva_ganancia_historico.csv",
        index=False,
    )

    ranking_base.to_csv(
        ventana_dir
        / "ranking_baseline.csv",
        index=False,
    )

    ranking_hist.to_csv(
        ventana_dir
        / "ranking_historico.csv",
        index=False,
    )

    comparacion = {
        "ventana":
            nombre,
        "train_months":
            train_months,
        "test_month":
            test_month,
        "seed":
            SEED,
        "baseline_n_features":
            len(
                features_originales
            ),
        "historico_n_features":
            len(
                features_historicas
            ),
        "baseline_auc":
            resultado_base[
                "auc"
            ],
        "historico_auc":
            resultado_hist[
                "auc"
            ],
        "delta_auc":
            delta_auc,
        "baseline_ap":
            resultado_base[
                "average_precision"
            ],
        "historico_ap":
            resultado_hist[
                "average_precision"
            ],
        "delta_ap":
            delta_ap,
        "baseline_gain_n12000_millones":
            resultado_base[
                "gain_n12000_millones"
            ],
        "historico_gain_n12000_millones":
            resultado_hist[
                "gain_n12000_millones"
            ],
        "delta_gain_n12000_millones":
            delta_gain_12000,
        "delta_gain_n12000_pct":
            delta_gain_12000_pct,
        "baseline_best_n":
            resultado_base[
                "best_n"
            ],
        "historico_best_n":
            resultado_hist[
                "best_n"
            ],
        "baseline_best_gain_millones":
            resultado_base[
                "best_gain_millones"
            ],
        "historico_best_gain_millones":
            resultado_hist[
                "best_gain_millones"
            ],
        "delta_best_gain_millones":
            delta_best_gain,
        "jaccard_top12000":
            jac,
        "interseccion_top12000":
            interseccion,
        "solo_baseline":
            solo_base,
        "solo_historico":
            solo_hist,
    }

    with (
        ventana_dir
        / "comparacion.json"
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
        "z524 - VALIDACIÓN FEATURE ENGINEERING HISTÓRICO",
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
        f"\nSeed: {SEED}",
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

    # Solo meses necesarios para este
    # experimento OOT.
    df = df.loc[
        df[
            MONTH_COL
        ].isin(
            [
                202104,
                202105,
                202106,
            ]
        )
    ].copy()

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
    # Validaciones estructurales
    # --------------------------------------------------------

    if len(
        features_originales
    ) != 152:
        raise ValueError(
            f"Esperábamos 152 "
            f"features originales; "
            f"hay "
            f"{len(features_originales)}"
        )

    if len(
        features_lag
    ) != 152:
        raise ValueError(
            f"Esperábamos 152 "
            f"features lag1; "
            f"hay "
            f"{len(features_lag)}"
        )

    if len(
        features_delta
    ) != 152:
        raise ValueError(
            f"Esperábamos 152 "
            f"features delta; "
            f"hay "
            f"{len(features_delta)}"
        )

    if len(
        features_historicas
    ) != 457:
        raise ValueError(
            f"Esperábamos 457 "
            f"features históricas; "
            f"hay "
            f"{len(features_historicas)}"
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

    print(
        "\nCobertura lag por mes:",
        flush=True,
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
        "mean"
    ] *= 100.0

    print(
        cobertura.to_string(),
        flush=True,
    )

    # --------------------------------------------------------
    # Ventanas
    # --------------------------------------------------------

    comparaciones = []

    for ventana in VENTANAS:

        resultado = evaluar_ventana(
            df=df,
            features_originales=
                features_originales,
            features_historicas=
                features_historicas,
            nombre=
                ventana[
                    "nombre"
                ],
            train_months=
                ventana[
                    "train_months"
                ],
            test_month=
                ventana[
                    "test_month"
                ],
        )

        comparaciones.append(
            resultado
        )

    # --------------------------------------------------------
    # Resumen final
    # --------------------------------------------------------

    resumen = pd.DataFrame(
        comparaciones
    )

    resumen.to_csv(
        OUTPUT_DIR
        / "resumen_z524.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "RESUMEN FINAL z524",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    cols = [
        "test_month",
        "baseline_auc",
        "historico_auc",
        "delta_auc",
        "baseline_gain_n12000_millones",
        "historico_gain_n12000_millones",
        "delta_gain_n12000_millones",
        "delta_gain_n12000_pct",
        "baseline_best_n",
        "historico_best_n",
        "baseline_best_gain_millones",
        "historico_best_gain_millones",
        "jaccard_top12000",
    ]

    print(
        "\n"
        + resumen[
            cols
        ].to_string(
            index=False,
            formatters={
                "baseline_auc":
                    lambda x:
                        f"{x:.6f}",
                "historico_auc":
                    lambda x:
                        f"{x:.6f}",
                "delta_auc":
                    lambda x:
                        f"{x:+.6f}",
                "baseline_gain_n12000_millones":
                    lambda x:
                        f"{x:.3f}",
                "historico_gain_n12000_millones":
                    lambda x:
                        f"{x:.3f}",
                "delta_gain_n12000_millones":
                    lambda x:
                        f"{x:+.3f}",
                "delta_gain_n12000_pct":
                    lambda x:
                        f"{x:+.2f}%",
                "baseline_best_gain_millones":
                    lambda x:
                        f"{x:.3f}",
                "historico_best_gain_millones":
                    lambda x:
                        f"{x:.3f}",
                "jaccard_top12000":
                    lambda x:
                        f"{x:.6f}",
            },
        ),
        flush=True,
    )

    # Señal simple, sin tomar decisión automática.
    mejoras = (
        resumen[
            "delta_gain_n12000_millones"
        ]
        > 0
    )

    print(
        "\nDiagnóstico N=12000:",
        flush=True,
    )

    print(
        f"  Ventanas con mejora: "
        f"{int(mejoras.sum())}"
        f"/{len(mejoras)}",
        flush=True,
    )

    print(
        f"  Delta medio: "
        f"{resumen['delta_gain_n12000_millones'].mean():+.3f} M",
        flush=True,
    )

    print(
        "\nNota:",
        flush=True,
    )

    print(
        "  La decisión no debe basarse "
        "solo en AUC ni en el mejor corte "
        "de cada ventana.",
        flush=True,
    )

    print(
        "  El contraste principal es "
        "baseline vs histórico a N=12000 "
        "sobre exactamente las mismas filas.",
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