#!/usr/bin/env python3
"""
z522_competencia_01_validacion_ensemble.py

Validación temporal del ensemble de semillas baseline usado en z521.

Objetivo
--------
Determinar si promediar las probabilidades de 5 modelos LightGBM que
difieren únicamente en random_state mejora o estabiliza el desempeño
fuera del Public Leaderboard.

Ventanas OOT:
    A) train 202103-202104 -> test 202105
    B) train 202103-202105 -> test 202106

Target:
    BAJA+2 = 1
    BAJA+1 / CONTINUA = 0

Para cada ventana:
    1. Entrena el baseline con 5 seeds.
    2. Calcula AUC, AP y logloss por seed.
    3. Calcula curva de ganancia por seed.
    4. Promedia las 5 probabilidades cliente por cliente.
    5. Calcula métricas y curva de ganancia del ensemble.
    6. Compara explícitamente:
         - gain a N=12000
         - mejor gain de cada ranking
         - estabilidad de los TOP-12000
         - Jaccard ensemble vs cada seed

IMPORTANTE
----------
No utiliza 202107.
No utiliza 202108.
No utiliza resultados del Public Leaderboard para ajustar el modelo.
Los hiperparámetros baseline quedan congelados.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
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
    "/data/dmeyf/datasets/competencia_01.csv"
)

OUTPUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "competencia_01/validacion_ensemble_z522"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"

SEEDS = [
    290497,
    100003,
    200003,
    300007,
    400009,
]

# Corte fijado en la estrategia final baseline/z521.
N_FIJO = 12000

# Misma grilla utilizada en las validaciones previas.
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
    "n_jobs": -1,
    "verbosity": -1,
}


VENTANAS = [
    {
        "nombre": "train_202103_202104_test_202105",
        "train_months": [
            202103,
            202104,
        ],
        "test_month": 202105,
    },
    {
        "nombre": "train_202103_202105_test_202106",
        "train_months": [
            202103,
            202104,
            202105,
        ],
        "test_month": 202106,
    },
]


# ============================================================
# Utilidades
# ============================================================

def calcular_ganancia(
    y_true: np.ndarray,
    indices_ordenados: np.ndarray,
    n: int,
) -> tuple[float, int, int]:

    idx = indices_ordenados[:n]

    positivos = int(
        y_true[idx].sum()
    )

    negativos = int(
        n - positivos
    )

    gain = (
        positivos
        * GANANCIA_ACIERTO
        + negativos
        * COSTO_ERROR
    )

    return (
        float(gain),
        positivos,
        negativos,
    )


def curva_ganancia(
    y_true: np.ndarray,
    probs: np.ndarray,
    cuts: list[int],
) -> pd.DataFrame:

    # Orden determinístico.
    order = np.argsort(
        -probs,
        kind="mergesort",
    )

    filas = []

    total_positivos = int(
        y_true.sum()
    )

    for n in cuts:

        if n > len(
            y_true
        ):
            continue

        gain, pos, neg = (
            calcular_ganancia(
                y_true=y_true,
                indices_ordenados=order,
                n=n,
            )
        )

        precision = (
            pos / n
            if n > 0
            else np.nan
        )

        recall = (
            pos / total_positivos
            if total_positivos > 0
            else np.nan
        )

        filas.append(
            {
                "n": n,
                "gain": gain,
                "gain_millones":
                    gain / 1_000_000,
                "positivos":
                    pos,
                "negativos":
                    neg,
                "precision":
                    precision,
                "recall":
                    recall,
            }
        )

    return pd.DataFrame(
        filas
    )


def metricas_clasificacion(
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


def obtener_fila_corte(
    curva: pd.DataFrame,
    n: int,
) -> pd.Series:

    fila = curva.loc[
        curva["n"] == n
    ]

    if len(
        fila
    ) != 1:
        raise ValueError(
            f"No se encontró exactamente "
            f"una fila para N={n}"
        )

    return fila.iloc[
        0
    ]


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
# Lectura de datos
# ============================================================

def cargar_datos():

    print(
        f"Leyendo esquema desde:\n"
        f"  {DATASET}",
        flush=True,
    )

    con = duckdb.connect()

    descripcion = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_csv_auto(
            '{DATASET}',
            header=true
        )
        """
    ).fetchdf()

    columnas = (
        descripcion[
            "column_name"
        ]
        .tolist()
    )

    excluir = {
        ID_COL,
        MONTH_COL,
        TARGET_COL,
    }

    features = [
        col
        for col in columnas
        if col not in excluir
    ]

    print(
        f"Columnas totales : "
        f"{len(columnas)}",
        flush=True,
    )

    print(
        f"Predictores      : "
        f"{len(features)}",
        flush=True,
    )

    if len(
        features
    ) != 152:
        raise ValueError(
            f"Se esperaban 152 "
            f"predictores y se "
            f"encontraron "
            f"{len(features)}"
        )

    meses_necesarios = [
        202103,
        202104,
        202105,
        202106,
    ]

    cols_sql = ", ".join(
        [
            f'"{ID_COL}"',
            f'"{MONTH_COL}"',
            f'"{TARGET_COL}"',
        ]
        + [
            f'"{c}"'
            for c in features
        ]
    )

    meses_sql = ", ".join(
        str(m)
        for m in meses_necesarios
    )

    print(
        "\nLeyendo meses "
        "202103-202106...",
        flush=True,
    )

    df = con.execute(
        f"""
        SELECT
            {cols_sql}
        FROM read_csv_auto(
            '{DATASET}',
            header=true
        )
        WHERE "{MONTH_COL}"
              IN ({meses_sql})
        """
    ).fetchdf()

    con.close()

    print(
        f"Filas cargadas   : "
        f"{len(df):,}",
        flush=True,
    )

    print(
        "\nDistribución por mes:",
        flush=True,
    )

    print(
        df[
            MONTH_COL
        ]
        .value_counts()
        .sort_index()
        .to_string(),
        flush=True,
    )

    if df[
        TARGET_COL
    ].isna().any():
        raise ValueError(
            "Hay targets NA en "
            "202103-202106"
        )

    # Target binario de competencia.
    df[
        "target_binario"
    ] = (
        df[
            TARGET_COL
        ]
        .eq(
            "BAJA+2"
        )
        .astype(
            np.int8
        )
    )

    return (
        df,
        features,
    )


# ============================================================
# Evaluación de una ventana
# ============================================================

def evaluar_ventana(
    df: pd.DataFrame,
    features: list[str],
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

    print(
        f"\nTrain rows: "
        f"{len(train):,}",
        flush=True,
    )

    print(
        f"Train BAJA+2: "
        f"{int(train['target_binario'].sum()):,}",
        flush=True,
    )

    print(
        f"Test rows : "
        f"{len(test):,}",
        flush=True,
    )

    print(
        f"Test BAJA+2: "
        f"{int(test['target_binario'].sum()):,}",
        flush=True,
    )

    X_train = train[
        features
    ]

    y_train = (
        train[
            "target_binario"
        ]
        .to_numpy()
    )

    X_test = test[
        features
    ]

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
    ) != len(
        ids_test
    ):
        raise ValueError(
            f"{nombre}: IDs duplicados "
            "en test"
        )

    probs_seeds = []

    resultados_seed = []

    curvas_seed = []

    tops_seed = {}

    ventana_dir = (
        OUTPUT_DIR
        / nombre
    )

    ventana_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Cinco modelos baseline
    # --------------------------------------------------------

    for i, seed in enumerate(
        SEEDS,
        start=1,
    ):

        print(
            "\n"
            + "-" * 80,
            flush=True,
        )

        print(
            f"Seed {seed} "
            f"({i}/{len(SEEDS)})",
            flush=True,
        )

        params = dict(
            BASE_PARAMS
        )

        params[
            "random_state"
        ] = seed

        model = LGBMClassifier(
            **params
        )

        t0 = time.time()

        model.fit(
            X_train,
            y_train,
        )

        probs = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        elapsed = (
            time.time()
            - t0
        )

        probs_seeds.append(
            probs
        )

        metricas = (
            metricas_clasificacion(
                y_true=y_test,
                probs=probs,
            )
        )

        curva = curva_ganancia(
            y_true=y_test,
            probs=probs,
            cuts=CUTS,
        )

        curva[
            "seed"
        ] = seed

        curvas_seed.append(
            curva
        )

        mejor = (
            curva.loc[
                curva[
                    "gain"
                ].idxmax()
            ]
        )

        fijo = (
            obtener_fila_corte(
                curva=curva,
                n=N_FIJO,
            )
        )

        order = np.argsort(
            -probs,
            kind="mergesort",
        )

        tops_seed[
            seed
        ] = set(
            ids_test[
                order[
                    :N_FIJO
                ]
            ].tolist()
        )

        resultados_seed.append(
            {
                "ventana":
                    nombre,
                "seed":
                    seed,
                "auc":
                    metricas[
                        "auc"
                    ],
                "average_precision":
                    metricas[
                        "average_precision"
                    ],
                "logloss":
                    metricas[
                        "logloss"
                    ],
                "best_n":
                    int(
                        mejor[
                            "n"
                        ]
                    ),
                "best_gain":
                    float(
                        mejor[
                            "gain"
                        ]
                    ),
                "best_gain_millones":
                    float(
                        mejor[
                            "gain_millones"
                        ]
                    ),
                "gain_n12000":
                    float(
                        fijo[
                            "gain"
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
                    elapsed,
            }
        )

        print(
            f"AUC      : "
            f"{metricas['auc']:.6f}",
            flush=True,
        )

        print(
            f"AP       : "
            f"{metricas['average_precision']:.6f}",
            flush=True,
        )

        print(
            f"LogLoss  : "
            f"{metricas['logloss']:.6f}",
            flush=True,
        )

        print(
            f"Best     : "
            f"N={int(mejor['n']):,} | "
            f"{mejor['gain_millones']:.3f} M",
            flush=True,
        )

        print(
            f"N=12.000 : "
            f"{fijo['gain_millones']:.3f} M | "
            f"positivos="
            f"{int(fijo['positivos'])}",
            flush=True,
        )

        print(
            f"Tiempo   : "
            f"{elapsed:.1f} s",
            flush=True,
        )

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 80,
        flush=True,
    )

    print(
        "ENSEMBLE 5 SEEDS",
        flush=True,
    )

    probs_matrix = np.column_stack(
        probs_seeds
    )

    probs_ensemble = (
        probs_matrix.mean(
            axis=1
        )
    )

    metricas_ensemble = (
        metricas_clasificacion(
            y_true=y_test,
            probs=probs_ensemble,
        )
    )

    curva_ensemble = (
        curva_ganancia(
            y_true=y_test,
            probs=probs_ensemble,
            cuts=CUTS,
        )
    )

    mejor_ensemble = (
        curva_ensemble.loc[
            curva_ensemble[
                "gain"
            ].idxmax()
        ]
    )

    fijo_ensemble = (
        obtener_fila_corte(
            curva=curva_ensemble,
            n=N_FIJO,
        )
    )

    order_ensemble = (
        np.argsort(
            -probs_ensemble,
            kind="mergesort",
        )
    )

    top_ensemble = set(
        ids_test[
            order_ensemble[
                :N_FIJO
            ]
        ].tolist()
    )

    # --------------------------------------------------------
    # Jaccard ensemble vs seeds
    # --------------------------------------------------------

    filas_jaccard = []

    for seed in SEEDS:

        top_seed = (
            tops_seed[
                seed
            ]
        )

        inter = len(
            top_ensemble
            & top_seed
        )

        union = len(
            top_ensemble
            | top_seed
        )

        filas_jaccard.append(
            {
                "seed":
                    seed,
                "interseccion":
                    inter,
                "union":
                    union,
                "jaccard":
                    jaccard(
                        top_ensemble,
                        top_seed,
                    ),
                "solo_ensemble":
                    len(
                        top_ensemble
                        - top_seed
                    ),
                "solo_seed":
                    len(
                        top_seed
                        - top_ensemble
                    ),
            }
        )

    jaccard_df = pd.DataFrame(
        filas_jaccard
    )

    # --------------------------------------------------------
    # Consenso dentro del TOP ensemble
    # --------------------------------------------------------

    consenso = []

    for cliente in top_ensemble:

        n_seeds = sum(
            cliente
            in tops_seed[
                seed
            ]
            for seed in SEEDS
        )

        consenso.append(
            n_seeds
        )

    consenso_counts = (
        pd.Series(
            consenso
        )
        .value_counts()
        .sort_index()
    )

    consenso_df = pd.DataFrame(
        {
            "n_seeds_top":
                consenso_counts.index,
            "clientes":
                consenso_counts.values,
        }
    )

    consenso_df[
        "porcentaje"
    ] = (
        100.0
        * consenso_df[
            "clientes"
        ]
        / N_FIJO
    )

    # --------------------------------------------------------
    # Comparación de ganancias
    # --------------------------------------------------------

    resultados_seed_df = (
        pd.DataFrame(
            resultados_seed
        )
    )

    gain_seed_mean = float(
        resultados_seed_df[
            "gain_n12000_millones"
        ].mean()
    )

    gain_seed_std = float(
        resultados_seed_df[
            "gain_n12000_millones"
        ].std(
            ddof=0
        )
    )

    gain_seed_min = float(
        resultados_seed_df[
            "gain_n12000_millones"
        ].min()
    )

    gain_seed_max = float(
        resultados_seed_df[
            "gain_n12000_millones"
        ].max()
    )

    gain_ensemble = float(
        fijo_ensemble[
            "gain_millones"
        ]
    )

    delta_vs_mean = (
        gain_ensemble
        - gain_seed_mean
    )

    delta_pct_vs_mean = (
        100.0
        * delta_vs_mean
        / gain_seed_mean
        if gain_seed_mean != 0
        else np.nan
    )

    # --------------------------------------------------------
    # Guardado
    # --------------------------------------------------------

    resultados_seed_df.to_csv(
        ventana_dir
        / "resultados_seeds.csv",
        index=False,
    )

    pd.concat(
        curvas_seed,
        ignore_index=True,
    ).to_csv(
        ventana_dir
        / "curvas_ganancia_seeds.csv",
        index=False,
    )

    curva_ensemble.to_csv(
        ventana_dir
        / "curva_ganancia_ensemble.csv",
        index=False,
    )

    jaccard_df.to_csv(
        ventana_dir
        / "jaccard_ensemble_vs_seeds.csv",
        index=False,
    )

    consenso_df.to_csv(
        ventana_dir
        / "consenso_top12000.csv",
        index=False,
    )

    # Ranking OOT completo para auditoría.
    ranking = pd.DataFrame(
        {
            ID_COL:
                ids_test,
            "target":
                y_test,
        }
    )

    for idx, seed in enumerate(
        SEEDS
    ):
        ranking[
            f"prob_{seed}"
        ] = probs_matrix[
            :,
            idx
        ]

    ranking[
        "prob_ensemble"
    ] = probs_ensemble

    ranking[
        "prob_std_seeds"
    ] = probs_matrix.std(
        axis=1,
        ddof=0,
    )

    ranking = (
        ranking
        .sort_values(
            "prob_ensemble",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    ranking[
        "rank_ensemble"
    ] = np.arange(
        1,
        len(ranking) + 1,
    )

    ranking.to_csv(
        ventana_dir
        / "ranking_oot_ensemble.csv",
        index=False,
    )

    resumen = {
        "ventana":
            nombre,
        "train_months":
            train_months,
        "test_month":
            test_month,
        "train_rows":
            len(train),
        "test_rows":
            len(test),
        "train_positives":
            int(
                y_train.sum()
            ),
        "test_positives":
            int(
                y_test.sum()
            ),
        "n_fijo":
            N_FIJO,
        "ensemble_auc":
            metricas_ensemble[
                "auc"
            ],
        "ensemble_average_precision":
            metricas_ensemble[
                "average_precision"
            ],
        "ensemble_logloss":
            metricas_ensemble[
                "logloss"
            ],
        "ensemble_best_n":
            int(
                mejor_ensemble[
                    "n"
                ]
            ),
        "ensemble_best_gain_millones":
            float(
                mejor_ensemble[
                    "gain_millones"
                ]
            ),
        "ensemble_gain_n12000_millones":
            gain_ensemble,
        "seed_gain_n12000_mean_millones":
            gain_seed_mean,
        "seed_gain_n12000_std_millones":
            gain_seed_std,
        "seed_gain_n12000_min_millones":
            gain_seed_min,
        "seed_gain_n12000_max_millones":
            gain_seed_max,
        "ensemble_delta_vs_seed_mean_millones":
            delta_vs_mean,
        "ensemble_delta_vs_seed_mean_pct":
            delta_pct_vs_mean,
        "jaccard_mean":
            float(
                jaccard_df[
                    "jaccard"
                ].mean()
            ),
        "jaccard_min":
            float(
                jaccard_df[
                    "jaccard"
                ].min()
            ),
        "jaccard_max":
            float(
                jaccard_df[
                    "jaccard"
                ].max()
            ),
    }

    with (
        ventana_dir
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

    # --------------------------------------------------------
    # Pantalla
    # --------------------------------------------------------

    print(
        f"AUC ensemble     : "
        f"{metricas_ensemble['auc']:.6f}",
        flush=True,
    )

    print(
        f"AP ensemble      : "
        f"{metricas_ensemble['average_precision']:.6f}",
        flush=True,
    )

    print(
        f"LogLoss ensemble : "
        f"{metricas_ensemble['logloss']:.6f}",
        flush=True,
    )

    print(
        f"Best ensemble    : "
        f"N={int(mejor_ensemble['n']):,} | "
        f"{mejor_ensemble['gain_millones']:.3f} M",
        flush=True,
    )

    print(
        f"Ensemble N=12000 : "
        f"{gain_ensemble:.3f} M | "
        f"positivos="
        f"{int(fijo_ensemble['positivos'])}",
        flush=True,
    )

    print(
        "\nGanancia N=12000 "
        "de semillas:",
        flush=True,
    )

    print(
        resultados_seed_df[
            [
                "seed",
                "gain_n12000_millones",
                "positivos_n12000",
            ]
        ].to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nResumen N=12000:",
        flush=True,
    )

    print(
        f"  seed mean : "
        f"{gain_seed_mean:.3f} M",
        flush=True,
    )

    print(
        f"  seed std  : "
        f"{gain_seed_std:.3f} M",
        flush=True,
    )

    print(
        f"  seed min  : "
        f"{gain_seed_min:.3f} M",
        flush=True,
    )

    print(
        f"  seed max  : "
        f"{gain_seed_max:.3f} M",
        flush=True,
    )

    print(
        f"  ensemble  : "
        f"{gain_ensemble:.3f} M",
        flush=True,
    )

    print(
        f"  delta     : "
        f"{delta_vs_mean:+.3f} M "
        f"({delta_pct_vs_mean:+.2f}%)",
        flush=True,
    )

    print(
        "\nJaccard ensemble "
        "vs seeds:",
        flush=True,
    )

    print(
        jaccard_df.to_string(
            index=False,
            formatters={
                "jaccard":
                    lambda x:
                        f"{x:.6f}"
            },
        ),
        flush=True,
    )

    print(
        "\nConsenso TOP-12000:",
        flush=True,
    )

    print(
        consenso_df.to_string(
            index=False,
            formatters={
                "porcentaje":
                    lambda x:
                        f"{x:.2f}%"
            },
        ),
        flush=True,
    )

    return resumen


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        "z522 - VALIDACIÓN TEMPORAL "
        "DEL ENSEMBLE BASELINE",
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

    df, features = (
        cargar_datos()
    )

    resumenes = []

    inicio = time.time()

    for ventana in VENTANAS:

        resumen = evaluar_ventana(
            df=df,
            features=features,
            nombre=ventana[
                "nombre"
            ],
            train_months=ventana[
                "train_months"
            ],
            test_month=ventana[
                "test_month"
            ],
        )

        resumenes.append(
            resumen
        )

    resumen_df = pd.DataFrame(
        resumenes
    )

    resumen_df.to_csv(
        OUTPUT_DIR
        / "resumen_ventanas_z522.csv",
        index=False,
    )

    elapsed_total = (
        time.time()
        - inicio
    )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "RESUMEN FINAL z522",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    columnas_resumen = [
        "test_month",
        "ensemble_gain_n12000_millones",
        "seed_gain_n12000_mean_millones",
        "seed_gain_n12000_std_millones",
        "ensemble_delta_vs_seed_mean_millones",
        "ensemble_delta_vs_seed_mean_pct",
        "ensemble_best_n",
        "ensemble_best_gain_millones",
        "jaccard_mean",
    ]

    print(
        "\n"
        + resumen_df[
            columnas_resumen
        ].to_string(
            index=False,
            formatters={
                "ensemble_gain_n12000_millones":
                    lambda x:
                        f"{x:.3f}",
                "seed_gain_n12000_mean_millones":
                    lambda x:
                        f"{x:.3f}",
                "seed_gain_n12000_std_millones":
                    lambda x:
                        f"{x:.3f}",
                "ensemble_delta_vs_seed_mean_millones":
                    lambda x:
                        f"{x:+.3f}",
                "ensemble_delta_vs_seed_mean_pct":
                    lambda x:
                        f"{x:+.2f}%",
                "ensemble_best_gain_millones":
                    lambda x:
                        f"{x:.3f}",
                "jaccard_mean":
                    lambda x:
                        f"{x:.6f}",
            },
        ),
        flush=True,
    )

    print(
        f"\nTiempo total de entrenamiento: "
        f"{elapsed_total / 60:.1f} min",
        flush=True,
    )

    print(
        f"\nResultados guardados en:\n"
        f"  {OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()