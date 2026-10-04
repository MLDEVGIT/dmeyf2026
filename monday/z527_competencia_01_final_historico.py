#!/usr/bin/env python3
"""
z527_competencia_01_final_historico.py

Candidato final histórico para Competencia 1.

Decisiones tomadas en z523-z526:
---------------------------------
- Target: BAJA+2.
- Feature Engineering histórico validado OOT.
- 457 features:
    152 originales
    152 lag1
    152 delta_lag1
      1 lag1_disponible
- Train supervisado:
    202104 + 202105 + 202106
- 202103 se excluye del entrenamiento histórico.
- 202107 NO se utiliza como target.
- 202107 solamente aporta información histórica para construir
  los lag1/delta_lag1 de 202108.
- Score:
    202108
- LightGBM:
    hiperparámetros baseline, sin tuning adicional.
- Seeds:
    290497, 100003, 200003, 300007, 400009
- Predicción final:
    promedio de probabilidades de las 5 seeds.
- Candidato principal:
    top 12.000 clientes.

El script:
-----------
1. Lee el parquet histórico construido por z523.
2. Audita train y score.
3. Entrena 5 LightGBM.
4. Guarda ranking completo por seed.
5. Calcula estabilidad entre seeds.
6. Promedia probabilidades.
7. Guarda ranking ensemble.
8. Genera CSV de submit sin header.
9. Valida estrictamente el CSV.
10. NO envía nada a Zulip.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier


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
    "competencia_01/submits_z527_historico"
)

ID_COL = "numero_de_cliente"
MONTH_COL = "foto_mes"
TARGET_COL = "clase_ternaria"
LAG_AVAILABLE_COL = "lag1_disponible"

TRAIN_MONTHS = [
    202104,
    202105,
    202106,
]

SCORE_MONTH = 202108

SEEDS = [
    290497,
    100003,
    200003,
    300007,
    400009,
]

N_PRINCIPAL = 12000

# Se guardan solamente como alternativas para análisis.
# No implica que debamos enviarlas al Public LB.
CORTES_ALTERNATIVOS = [
    8000,
    9500,
    12000,
    12500,
    13500,
    14000,
    14500,
    15000,
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
}


# ============================================================
# Utilidades
# ============================================================

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


def top_n_ids(
    ids: np.ndarray,
    probs: np.ndarray,
    n: int,
) -> set[int]:

    order = np.lexsort(
        (
            ids,
            -probs,
        )
    )

    return set(
        ids[
            order[:n]
        ].tolist()
    )


def ranking_dataframe(
    ids: np.ndarray,
    probs: np.ndarray,
) -> pd.DataFrame:

    ranking = pd.DataFrame(
        {
            ID_COL: ids,
            "prob": probs,
        }
    )

    ranking = (
        ranking.sort_values(
            [
                "prob",
                ID_COL,
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
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
        dtype=np.int64,
    )

    return ranking


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
# Validación del submit
# ============================================================

def validar_submit(
    path: Path,
    ids_score: np.ndarray,
    n_esperado: int,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"No existe submit: {path}"
        )

    submit = pd.read_csv(
        path,
        header=None,
        names=[
            ID_COL
        ],
    )

    errores = []

    if len(
        submit
    ) != n_esperado:

        errores.append(
            f"Filas: {len(submit):,}; "
            f"esperadas: {n_esperado:,}"
        )

    if submit[
        ID_COL
    ].isna().any():

        errores.append(
            "Hay IDs nulos."
        )

    numeric = pd.to_numeric(
        submit[
            ID_COL
        ],
        errors="coerce",
    )

    if numeric.isna().any():

        errores.append(
            "Hay IDs no numéricos."
        )

    else:

        valores = numeric.to_numpy(
            dtype=np.float64
        )

        if not np.all(
            np.equal(
                valores,
                np.floor(
                    valores
                ),
            )
        ):

            errores.append(
                "Hay IDs no enteros."
            )

        ids_submit = set(
            numeric.astype(
                np.int64
            ).tolist()
        )

        if len(
            ids_submit
        ) != n_esperado:

            errores.append(
                "Hay IDs duplicados."
            )

        ids_score_set = set(
            ids_score.tolist()
        )

        fuera_score = (
            ids_submit
            - ids_score_set
        )

        if fuera_score:

            errores.append(
                f"Hay {len(fuera_score):,} "
                "IDs que no pertenecen "
                "a 202108."
            )

    # Verificación textual:
    # exactamente una columna y sin notación científica.
    lineas = (
        path.read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    if len(
        lineas
    ) != n_esperado:

        errores.append(
            "Cantidad de líneas textual "
            "incorrecta."
        )

    lineas_invalidas = []

    for i, linea in enumerate(
        lineas,
        start=1,
    ):

        texto = linea.strip()

        if (
            not texto
            or not texto.isdigit()
        ):

            lineas_invalidas.append(
                i
            )

            if len(
                lineas_invalidas
            ) >= 10:
                break

    if lineas_invalidas:

        errores.append(
            "Hay líneas que no son "
            "enteros puros. Primeras: "
            f"{lineas_invalidas}"
        )

    if errores:

        raise ValueError(
            "\n".join(
                errores
            )
        )

    return {
        "path": str(
            path
        ),
        "filas": int(
            len(submit)
        ),
        "ids_unicos": int(
            submit[
                ID_COL
            ].nunique()
        ),
        "valido": True,
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
        "z527 - CANDIDATO FINAL HISTÓRICO",
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
        f"\nTrain: {TRAIN_MONTHS}",
        flush=True,
    )

    print(
        f"Score: {SCORE_MONTH}",
        flush=True,
    )

    print(
        f"Seeds: {SEEDS}",
        flush=True,
    )

    print(
        f"Candidato principal: "
        f"N={N_PRINCIPAL:,}",
        flush=True,
    )

    # ========================================================
    # Lectura
    # ========================================================

    print(
        "\nLeyendo parquet histórico...",
        flush=True,
    )

    df = pd.read_parquet(
        DATASET
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
        f"\nFilas dataset        : "
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

    # ========================================================
    # Auditorías de esquema
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
        features_historicas
    ) != 457:

        raise ValueError(
            "Se esperaban 457 "
            "features históricas."
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
    # Train / Score
    # ========================================================

    train = (
        df.loc[
            df[
                MONTH_COL
            ].isin(
                TRAIN_MONTHS
            )
        ]
        .copy()
    )

    score = (
        df.loc[
            df[
                MONTH_COL
            ].eq(
                SCORE_MONTH
            )
        ]
        .copy()
    )

    if len(
        train
    ) == 0:

        raise ValueError(
            "Train vacío."
        )

    if len(
        score
    ) == 0:

        raise ValueError(
            "Score vacío."
        )

    if (
        train[
            TARGET_COL
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Hay targets NA en train."
        )

    # En competencia 202108 debe estar sin target.
    score_target_no_na = int(
        score[
            TARGET_COL
        ]
        .notna()
        .sum()
    )

    print(
        f"\nTrain rows       : "
        f"{len(train):,}",
        flush=True,
    )

    print(
        f"Score rows       : "
        f"{len(score):,}",
        flush=True,
    )

    print(
        f"Target no-NA score: "
        f"{score_target_no_na:,}",
        flush=True,
    )

    if score_target_no_na != 0:

        raise ValueError(
            "202108 contiene target conocido. "
            "Revisar dataset."
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

    ids_score = (
        score[
            ID_COL
        ]
        .astype(
            np.int64
        )
        .to_numpy()
    )

    if len(
        np.unique(
            ids_score
        )
    ) != len(
        ids_score
    ):

        raise ValueError(
            "Hay IDs duplicados en 202108."
        )

    print(
        f"Train BAJA+2     : "
        f"{int(y_train.sum()):,}",
        flush=True,
    )

    print(
        f"Train no BAJA+2  : "
        f"{int((y_train == 0).sum()):,}",
        flush=True,
    )

    print(
        f"Score IDs únicos : "
        f"{len(np.unique(ids_score)):,}",
        flush=True,
    )

    # ========================================================
    # Cobertura histórica
    # ========================================================

    print(
        "\nCobertura lag en train:",
        flush=True,
    )

    cobertura_train = (
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

    cobertura_train[
        "pct"
    ] = (
        cobertura_train[
            "mean"
        ]
        * 100.0
    )

    print(
        cobertura_train[
            [
                "count",
                "sum",
                "pct",
            ]
        ].to_string(),
        flush=True,
    )

    score_lag_count = int(
        score[
            LAG_AVAILABLE_COL
        ].sum()
    )

    score_lag_pct = (
        100.0
        * score_lag_count
        / len(score)
    )

    print(
        f"\nCobertura lag 202108: "
        f"{score_lag_count:,}/"
        f"{len(score):,} "
        f"({score_lag_pct:.4f}%)",
        flush=True,
    )

    # Esperamos aproximadamente 99%.
    if score_lag_pct < 95.0:

        raise ValueError(
            "Cobertura histórica de 202108 "
            "inesperadamente baja."
        )

    # ========================================================
    # Matrices
    # ========================================================

    X_train = train[
        features_historicas
    ]

    X_score = score[
        features_historicas
    ]

    print(
        f"\nX_train: "
        f"{X_train.shape}",
        flush=True,
    )

    print(
        f"X_score: "
        f"{X_score.shape}",
        flush=True,
    )

    # ========================================================
    # Entrenamiento de seeds
    # ========================================================

    probabilidades = {}
    tops_principales = {}
    resumen_seeds = []

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
            f"[{i}/{len(SEEDS)}] "
            f"Seed {seed}",
            flush=True,
        )

        print(
            "-" * 80,
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
                X_score
            )[:, 1]
        )

        elapsed = (
            time.time()
            - t0
        )

        if not np.all(
            np.isfinite(
                probs
            )
        ):

            raise ValueError(
                f"Seed {seed}: "
                "probabilidades no finitas."
            )

        if (
            np.min(
                probs
            ) < 0.0
            or np.max(
                probs
            ) > 1.0
        ):

            raise ValueError(
                f"Seed {seed}: "
                "probabilidades fuera "
                "de [0,1]."
            )

        probabilidades[
            seed
        ] = probs

        ranking = (
            ranking_dataframe(
                ids=ids_score,
                probs=probs,
            )
        )

        ranking_path = (
            OUTPUT_DIR
            / (
                f"ranking_historico_"
                f"seed{seed}.csv"
            )
        )

        ranking.to_csv(
            ranking_path,
            index=False,
        )

        tops_principales[
            seed
        ] = set(
            ranking.iloc[
                :N_PRINCIPAL
            ][
                ID_COL
            ]
            .astype(
                np.int64
            )
            .tolist()
        )

        resumen_seed = {
            "seed": seed,
            "elapsed_seconds":
                float(
                    elapsed
                ),
            "prob_min":
                float(
                    np.min(
                        probs
                    )
                ),
            "prob_mean":
                float(
                    np.mean(
                        probs
                    )
                ),
            "prob_max":
                float(
                    np.max(
                        probs
                    )
                ),
            "ranking_path":
                str(
                    ranking_path
                ),
        }

        resumen_seeds.append(
            resumen_seed
        )

        print(
            f"Tiempo    : "
            f"{elapsed:.1f} s",
            flush=True,
        )

        print(
            f"Prob min  : "
            f"{np.min(probs):.8f}",
            flush=True,
        )

        print(
            f"Prob media: "
            f"{np.mean(probs):.8f}",
            flush=True,
        )

        print(
            f"Prob max  : "
            f"{np.max(probs):.8f}",
            flush=True,
        )

        print(
            f"Ranking   : "
            f"{ranking_path}",
            flush=True,
        )

    resumen_seeds_df = pd.DataFrame(
        resumen_seeds
    )

    resumen_seeds_df.to_csv(
        OUTPUT_DIR
        / "resumen_seeds.csv",
        index=False,
    )

    # ========================================================
    # Estabilidad entre seeds
    # ========================================================

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        f"ESTABILIDAD TOP {N_PRINCIPAL:,} "
        "ENTRE SEEDS",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    jaccard_rows = []

    for (
        seed_a,
        seed_b,
    ) in itertools.combinations(
        SEEDS,
        2,
    ):

        top_a = (
            tops_principales[
                seed_a
            ]
        )

        top_b = (
            tops_principales[
                seed_b
            ]
        )

        jac = jaccard(
            top_a,
            top_b,
        )

        interseccion = len(
            top_a
            & top_b
        )

        jaccard_rows.append(
            {
                "seed_a":
                    seed_a,
                "seed_b":
                    seed_b,
                "jaccard":
                    jac,
                "interseccion":
                    interseccion,
            }
        )

        print(
            f"{seed_a} vs {seed_b}: "
            f"Jaccard={jac:.6f} "
            f"| intersección="
            f"{interseccion:,}",
            flush=True,
        )

    jaccard_df = pd.DataFrame(
        jaccard_rows
    )

    jaccard_df.to_csv(
        OUTPUT_DIR
        / "jaccard_seeds_n12000.csv",
        index=False,
    )

    print(
        f"\nJaccard medio seeds: "
        f"{jaccard_df['jaccard'].mean():.6f}",
        flush=True,
    )

    print(
        f"Jaccard mínimo     : "
        f"{jaccard_df['jaccard'].min():.6f}",
        flush=True,
    )

    print(
        f"Jaccard máximo     : "
        f"{jaccard_df['jaccard'].max():.6f}",
        flush=True,
    )

    # ========================================================
    # Ensemble
    # ========================================================

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "ENSEMBLE 5 SEEDS",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

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

    ranking_ensemble = (
        ranking_dataframe(
            ids=ids_score,
            probs=probs_ensemble,
        )
    )

    ranking_ensemble_path = (
        OUTPUT_DIR
        / "ranking_historico_ensemble5.csv"
    )

    ranking_ensemble.to_csv(
        ranking_ensemble_path,
        index=False,
    )

    print(
        f"Prob ensemble min  : "
        f"{np.min(probs_ensemble):.8f}",
        flush=True,
    )

    print(
        f"Prob ensemble media: "
        f"{np.mean(probs_ensemble):.8f}",
        flush=True,
    )

    print(
        f"Prob ensemble max  : "
        f"{np.max(probs_ensemble):.8f}",
        flush=True,
    )

    print(
        f"Ranking ensemble   : "
        f"{ranking_ensemble_path}",
        flush=True,
    )

    # ========================================================
    # Ensemble vs seeds
    # ========================================================

    top_ensemble_principal = set(
        ranking_ensemble.iloc[
            :N_PRINCIPAL
        ][
            ID_COL
        ]
        .astype(
            np.int64
        )
        .tolist()
    )

    ensemble_vs_seed_rows = []

    print(
        "\nEnsemble vs seeds:",
        flush=True,
    )

    for seed in SEEDS:

        top_seed = (
            tops_principales[
                seed
            ]
        )

        jac = jaccard(
            top_ensemble_principal,
            top_seed,
        )

        interseccion = len(
            top_ensemble_principal
            & top_seed
        )

        ensemble_vs_seed_rows.append(
            {
                "seed":
                    seed,
                "jaccard":
                    jac,
                "interseccion":
                    interseccion,
            }
        )

        print(
            f"  seed {seed}: "
            f"Jaccard={jac:.6f} "
            f"| intersección="
            f"{interseccion:,}",
            flush=True,
        )

    ensemble_vs_seed_df = pd.DataFrame(
        ensemble_vs_seed_rows
    )

    ensemble_vs_seed_df.to_csv(
        OUTPUT_DIR
        / "jaccard_ensemble_vs_seeds.csv",
        index=False,
    )

    print(
        f"\nJaccard ensemble medio: "
        f"{ensemble_vs_seed_df['jaccard'].mean():.6f}",
        flush=True,
    )

    # ========================================================
    # Consenso de seeds dentro del top ensemble
    # ========================================================

    contador_consenso = {}

    for seed in SEEDS:

        for cliente in (
            tops_principales[
                seed
            ]
        ):

            contador_consenso[
                cliente
            ] = (
                contador_consenso.get(
                    cliente,
                    0,
                )
                + 1
            )

    consenso_rows = []

    print(
        "\nConsenso dentro del "
        "top 12.000 ensemble:",
        flush=True,
    )

    for cantidad_seeds in range(
        1,
        len(SEEDS) + 1,
    ):

        cantidad_clientes = sum(
            1
            for cliente
            in top_ensemble_principal
            if contador_consenso.get(
                cliente,
                0,
            )
            == cantidad_seeds
        )

        porcentaje = (
            100.0
            * cantidad_clientes
            / N_PRINCIPAL
        )

        consenso_rows.append(
            {
                "cantidad_seeds":
                    cantidad_seeds,
                "clientes":
                    cantidad_clientes,
                "porcentaje":
                    porcentaje,
            }
        )

        print(
            f"  {cantidad_seeds}/5 seeds: "
            f"{cantidad_clientes:,} "
            f"({porcentaje:.2f}%)",
            flush=True,
        )

    consenso_df = pd.DataFrame(
        consenso_rows
    )

    consenso_df.to_csv(
        OUTPUT_DIR
        / "consenso_top12000.csv",
        index=False,
    )

    # ========================================================
    # Generación de submits
    # ========================================================

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "GENERACIÓN DE SUBMITS",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    submits = []

    for n in CORTES_ALTERNATIVOS:

        if n > len(
            ranking_ensemble
        ):

            raise ValueError(
                f"N={n:,} supera "
                "cantidad de score."
            )

        submit = (
            ranking_ensemble.iloc[
                :n
            ][
                [
                    ID_COL
                ]
            ]
            .astype(
                np.int64
            )
        )

        submit_path = (
            OUTPUT_DIR
            / (
                "submit_historico_"
                "ensemble5seeds_"
                f"n{n}.csv"
            )
        )

        submit.to_csv(
            submit_path,
            index=False,
            header=False,
        )

        validacion = (
            validar_submit(
                path=submit_path,
                ids_score=ids_score,
                n_esperado=n,
            )
        )

        submits.append(
            {
                "n":
                    n,
                "path":
                    str(
                        submit_path
                    ),
                "valido":
                    validacion[
                        "valido"
                    ],
            }
        )

        marca = (
            "  <-- PRINCIPAL"
            if n
            == N_PRINCIPAL
            else ""
        )

        print(
            f"N={n:>5,}: "
            f"OK | "
            f"{submit_path.name}"
            f"{marca}",
            flush=True,
        )

    submits_df = pd.DataFrame(
        submits
    )

    submits_df.to_csv(
        OUTPUT_DIR
        / "submits_generados.csv",
        index=False,
    )

    # ========================================================
    # Validación final específica del candidato principal
    # ========================================================

    principal_path = (
        OUTPUT_DIR
        / (
            "submit_historico_"
            "ensemble5seeds_"
            f"n{N_PRINCIPAL}.csv"
        )
    )

    principal = pd.read_csv(
        principal_path,
        header=None,
        names=[
            ID_COL
        ],
    )

    if len(
        principal
    ) != N_PRINCIPAL:

        raise ValueError(
            "Submit principal no tiene "
            "12.000 filas."
        )

    if (
        principal[
            ID_COL
        ].nunique()
        != N_PRINCIPAL
    ):

        raise ValueError(
            "Submit principal tiene "
            "IDs duplicados."
        )

    # Debe coincidir exactamente con top ensemble.
    principal_set = set(
        principal[
            ID_COL
        ]
        .astype(
            np.int64
        )
        .tolist()
    )

    if (
        principal_set
        != top_ensemble_principal
    ):

        raise ValueError(
            "Submit principal no coincide "
            "con top 12.000 ensemble."
        )

    # ========================================================
    # Metadata reproducible
    # ========================================================

    metadata = {
        "script":
            "z527_competencia_01_final_historico.py",

        "dataset":
            str(
                DATASET
            ),

        "train_months":
            TRAIN_MONTHS,

        "score_month":
            SCORE_MONTH,

        "target":
            "BAJA+2",

        "n_features_originales":
            len(
                features_originales
            ),

        "n_features_lag1":
            len(
                features_lag
            ),

        "n_features_delta_lag1":
            len(
                features_delta
            ),

        "n_features_total":
            len(
                features_historicas
            ),

        "train_rows":
            int(
                len(train)
            ),

        "train_positivos":
            int(
                y_train.sum()
            ),

        "score_rows":
            int(
                len(score)
            ),

        "score_lag_count":
            score_lag_count,

        "score_lag_pct":
            score_lag_pct,

        "seeds":
            SEEDS,

        "params":
            PARAMS_BASE,

        "n_principal":
            N_PRINCIPAL,

        "cortes_generados":
            CORTES_ALTERNATIVOS,

        "jaccard_seed_mean":
            float(
                jaccard_df[
                    "jaccard"
                ].mean()
            ),

        "jaccard_seed_min":
            float(
                jaccard_df[
                    "jaccard"
                ].min()
            ),

        "jaccard_seed_max":
            float(
                jaccard_df[
                    "jaccard"
                ].max()
            ),

        "jaccard_ensemble_seed_mean":
            float(
                ensemble_vs_seed_df[
                    "jaccard"
                ].mean()
            ),

        "principal_submit":
            str(
                principal_path
            ),
    }

    with (
        OUTPUT_DIR
        / "metadata_z527.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # Resumen final
    # ========================================================

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
        "RESUMEN FINAL z527",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"\nTrain supervisado : "
        f"{TRAIN_MONTHS}",
        flush=True,
    )

    print(
        f"Train rows        : "
        f"{len(train):,}",
        flush=True,
    )

    print(
        f"Train BAJA+2      : "
        f"{int(y_train.sum()):,}",
        flush=True,
    )

    print(
        f"Score             : "
        f"{SCORE_MONTH}",
        flush=True,
    )

    print(
        f"Score rows        : "
        f"{len(score):,}",
        flush=True,
    )

    print(
        f"Features          : "
        f"{len(features_historicas):,}",
        flush=True,
    )

    print(
        f"Seeds             : "
        f"{len(SEEDS)}",
        flush=True,
    )

    print(
        f"Lag disponible    : "
        f"{score_lag_pct:.4f}%",
        flush=True,
    )

    print(
        f"Jaccard seeds     : "
        f"{jaccard_df['jaccard'].mean():.6f}",
        flush=True,
    )

    print(
        f"Jaccard ens/seeds : "
        f"{ensemble_vs_seed_df['jaccard'].mean():.6f}",
        flush=True,
    )

    print(
        "\nCANDIDATO PRINCIPAL:",
        flush=True,
    )

    print(
        f"  N={N_PRINCIPAL:,}",
        flush=True,
    )

    print(
        f"  {principal_path}",
        flush=True,
    )

    print(
        "\nValidación:",
        flush=True,
    )

    print(
        f"  filas       : "
        f"{len(principal):,}",
        flush=True,
    )

    print(
        f"  IDs únicos  : "
        f"{principal[ID_COL].nunique():,}",
        flush=True,
    )

    print(
        "  header      : NO",
        flush=True,
    )

    print(
        "  IDs enteros : SI",
        flush=True,
    )

    print(
        "  score 202108: SI",
        flush=True,
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
        "\nIMPORTANTE: "
        "este script NO envió nada a Zulip.",
        flush=True,
    )


if __name__ == "__main__":
    main()