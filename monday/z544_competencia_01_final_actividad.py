#!/usr/bin/env python3
"""
z544_competencia_01_final_actividad.py

Candidatos finales ACTIVIDAD para Competencia 1.

Decisión experimental
---------------------
Los experimentos z540-z543 evaluaron una feature de actividad
transaccional construida como:

    actividad_canales =
        cantidad de variables transaccionales con valor > 0

La ablación, robustez temporal y robustez por seed justifican
llevar FULL457 + actividad_canales al scoring final.

Configuración final
-------------------
Train supervisado:
    202104 + 202105 + 202106

Score:
    202108

Features:
    152 originales
    152 lag1
    152 delta_lag1
      1 lag1_disponible
      1 actividad_canales
    ------------------
    458 features

Target:
    BAJA+2

Modelo:
    LightGBM con parámetros establecidos previamente

Seeds:
    290497
    100003
    200003
    300007
    400009

Candidatos:
    1. seed canónica 290497
    2. ensemble promedio de las 5 seeds

Cortes:
    8000
    8500
    9000
    9500
    12000

IMPORTANTE
----------
- 202103 no se usa para entrenamiento.
- 202107 no se usa como target.
- 202107 puede aportar historia a las features de 202108.
- 202108 se utiliza exclusivamente para scoring.
- No se seleccionan parámetros mirando 202108.
- No utiliza el Public LB para seleccionar parámetros.
- No envía nada a Zulip.
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
    "competencia_01/submits_z544_actividad"
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

N_PRINCIPAL = 8000

CORTES_ALTERNATIVOS = [
    8000,
    8500,
    9000,
    9500,
    12000,
]


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

    return len(a & b) / len(union)


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

    ranking["ranking"] = np.arange(
        1,
        len(ranking) + 1,
        dtype=np.int64,
    )

    return ranking


# ============================================================
# Identificación estricta de features
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
        "lag2_disponible",
        "actividad_canales",
    }

    features_delta = [
        c
        for c in df.columns
        if c.endswith("_delta_lag1")
    ]

    features_lag1 = [
        c
        for c in df.columns
        if (
            c.endswith("_lag1")
            and not c.endswith("_delta_lag1")
        )
    ]

    features_lag2 = [
        c
        for c in df.columns
        if c.endswith("_lag2")
    ]

    features_originales = [
        c
        for c in df.columns
        if (
            c not in excluir
            and not c.endswith("_lag1")
            and not c.endswith("_delta_lag1")
            and not c.endswith("_lag2")
        )
    ]

    features_full457 = (
        features_originales
        + features_lag1
        + features_delta
        + [LAG_AVAILABLE_COL]
    )

    features_actual458 = (
        features_full457
        + ["actividad_canales"]
    )

    return (
        features_originales,
        features_lag1,
        features_delta,
        features_actual458,
    )


# ============================================================
# Validación estricta del submit
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

    if len(submit) != n_esperado:

        errores.append(
            f"Filas={len(submit):,}; "
            f"esperadas={n_esperado:,}"
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
            valores
            == np.floor(
                valores
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
                f"{len(fuera_score):,} IDs "
                "no pertenecen a 202108."
            )

    # Auditoría textual.
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
            "Cantidad textual de líneas "
            "incorrecta."
        )

    invalidas = []

    for i, linea in enumerate(
        lineas,
        start=1,
    ):

        texto = linea.strip()

        if (
            not texto
            or not texto.isdigit()
        ):

            invalidas.append(
                i
            )

            if len(
                invalidas
            ) >= 10:
                break

    if invalidas:

        errores.append(
            "Líneas no enteras. "
            f"Primeras: {invalidas}"
        )

    if errores:

        raise ValueError(
            "\n".join(
                errores
            )
        )

    return {
        "path":
            str(path),

        "filas":
            int(
                len(submit)
            ),

        "ids_unicos":
            int(
                submit[
                    ID_COL
                ].nunique()
            ),

        "valido":
            True,
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
        "z544 - CANDIDATO FINAL ACTIVIDAD",
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

    # ========================================================
    # Feature de actividad
    # ========================================================

    faltantes = [
        c
        for c in TRANSACTION_VARS
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan variables transaccionales: "
            f"{faltantes}"
        )

    df["actividad_canales"] = (
        df[TRANSACTION_VARS]
        .gt(0)
        .sum(axis=1)
        .astype(np.int16)
    )

    (
        features_originales,
        features_lag,
        features_delta,
        features_modelo,
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
        f"Features ACTUAL modelo: "
        f"{len(features_modelo):,}",
        flush=True,
    )

    # ========================================================
    # Auditorías del esquema ACTUAL458
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

    if len(features_modelo) != 458:
        raise ValueError(
            "El modelo ACTUAL debe tener "
            "458 features."
        )

    if features_modelo.count(
        "actividad_canales"
    ) != 1:
        raise ValueError(
            "actividad_canales debe aparecer "
            "exactamente una vez."
        )

    if LAG_AVAILABLE_COL not in features_modelo:
        raise ValueError(
            "Falta lag1_disponible "
            "en ACTUAL458."
        )

    # ACTUAL458 no utiliza features lag2;
    # esta auditoría evita su ingreso accidental.
    lag2_en_modelo = [
        c
        for c in features_modelo
        if (
            c.endswith("_lag2")
            or c == "lag2_disponible"
        )
    ]

    if lag2_en_modelo:
        raise ValueError(
            "Se detectaron features lag2 "
            "en ACTUAL458: "
            f"{lag2_en_modelo[:10]}"
        )

    # Auditoría de duplicados de features.
    if len(features_modelo) != len(
        set(features_modelo)
    ):
        raise ValueError(
            "Hay features duplicadas "
            "en ACTUAL458."
        )

    print(
        "\nEsquema ACTUAL458 validado:",
        flush=True,
    )

    print(
        "  originales       : 152",
        flush=True,
    )

    print(
        "  lag1 crudos      : 152",
        flush=True,
    )

    print(
        "  delta_lag1       : 152",
        flush=True,
    )

    print(
        "  lag1_disponible  : 1",
        flush=True,
    )

    print(
        "  actividad_canales: 1",
        flush=True,
    )

    print(
        "  TOTAL            : 458",
        flush=True,
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
    # Train / score
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

    if len(train) == 0:

        raise ValueError(
            "Train vacío."
        )

    if len(score) == 0:

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

    score_target_no_na = int(
        score[
            TARGET_COL
        ]
        .notna()
        .sum()
    )

    if score_target_no_na != 0:

        raise ValueError(
            "202108 contiene target "
            "conocido."
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
            "Hay IDs duplicados "
            "en score."
        )

    print(
        f"\nTrain rows       : "
        f"{len(train):,}",
        flush=True,
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
        f"Score rows       : "
        f"{len(score):,}",
        flush=True,
    )

    print(
        f"Score IDs únicos : "
        f"{len(np.unique(ids_score)):,}",
        flush=True,
    )

    print(
        f"Target no-NA score: "
        f"{score_target_no_na:,}",
        flush=True,
    )

    # ========================================================
    # Cobertura histórica
    # ========================================================

    print(
        "\nCobertura histórica train:",
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
        f"\nCobertura histórica 202108: "
        f"{score_lag_count:,}/"
        f"{len(score):,} "
        f"({score_lag_pct:.4f}%)",
        flush=True,
    )

    if score_lag_pct < 95.0:

        raise ValueError(
            "Cobertura histórica "
            "inesperadamente baja."
        )

    # ========================================================
    # Matrices
    # ========================================================

    X_train = train[
        features_modelo
    ]

    X_score = score[
        features_modelo
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
    # Entrenamiento
    # ========================================================

    probabilidades = {}
    tops = {}
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
            np.min(probs) < 0.0
            or np.max(probs) > 1.0
        ):

            raise ValueError(
                f"Seed {seed}: "
                "probabilidades fuera "
                "de [0,1]."
            )

        probabilidades[
            seed
        ] = probs

        ranking = ranking_dataframe(
            ids=ids_score,
            probs=probs,
        )

        ranking_path = (
            OUTPUT_DIR
            / (
                "ranking_actividad_"
                f"seed{seed}.csv"
            )
        )

        ranking.to_csv(
            ranking_path,
            index=False,
        )

        tops[
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

        resumen_seeds.append(
            {
                "seed":
                    seed,

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

    pd.DataFrame(
        resumen_seeds
    ).to_csv(
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
        f"ESTABILIDAD TOP "
        f"{N_PRINCIPAL:,} ENTRE SEEDS",
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

        top_a = tops[
            seed_a
        ]

        top_b = tops[
            seed_b
        ]

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
        / "jaccard_seeds_n8000.csv",
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
        "ENSEMBLE ACTIVIDAD - 5 SEEDS",
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
        / "ranking_actividad_ensemble5.csv"
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

    top_ensemble = set(
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

    # ========================================================
    # Ensemble vs seeds
    # ========================================================

    ensemble_vs_seed_rows = []

    print(
        "\nEnsemble vs seeds:",
        flush=True,
    )

    for seed in SEEDS:

        jac = jaccard(
            top_ensemble,
            tops[
                seed
            ],
        )

        interseccion = len(
            top_ensemble
            & tops[
                seed
            ]
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
    # Consenso
    # ========================================================

    contador_consenso = {}

    for seed in SEEDS:

        for cliente in tops[
            seed
        ]:

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
        f"top {N_PRINCIPAL:,} ensemble:",
        flush=True,
    )

    for cantidad_seeds in range(
        1,
        len(SEEDS) + 1,
    ):

        cantidad = sum(
            1
            for cliente
            in top_ensemble
            if contador_consenso.get(
                cliente,
                0,
            )
            == cantidad_seeds
        )

        pct = (
            100.0
            * cantidad
            / N_PRINCIPAL
        )

        consenso_rows.append(
            {
                "cantidad_seeds":
                    cantidad_seeds,

                "clientes":
                    cantidad,

                "porcentaje":
                    pct,
            }
        )

        print(
            f"  {cantidad_seeds}/5 seeds: "
            f"{cantidad:,} "
            f"({pct:.2f}%)",
            flush=True,
        )

    pd.DataFrame(
        consenso_rows
    ).to_csv(
        OUTPUT_DIR
        / "consenso_top8000.csv",
        index=False,
    )

    # ========================================================
    # Candidatos finales
    # ========================================================

    CANONICAL_SEED = 290497

    ranking_canonical = ranking_dataframe(
        ids=ids_score,
        probs=probabilidades[CANONICAL_SEED],
    )

    ranking_canonical_path = (
        OUTPUT_DIR
        / (
            "ranking_actividad_"
            f"seed{CANONICAL_SEED}_final.csv"
        )
    )

    ranking_canonical.to_csv(
        ranking_canonical_path,
        index=False,
    )

    # Comparación de rankings finales.
    top_canonical = set(
        ranking_canonical.iloc[
            :N_PRINCIPAL
        ][ID_COL]
        .astype(np.int64)
        .tolist()
    )

    inter_final = len(
        top_canonical
        & top_ensemble
    )

    union_final = len(
        top_canonical
        | top_ensemble
    )

    jaccard_final = (
        inter_final / union_final
    )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        "CANDIDATOS FINALES 202108",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    print(
        f"Canonical seed : {CANONICAL_SEED}",
        flush=True,
    )

    print(
        "Ensemble       : promedio 5 seeds",
        flush=True,
    )

    print(
        f"Top {N_PRINCIPAL:,} - "
        f"intersección   : {inter_final:,}",
        flush=True,
    )

    print(
        f"Top {N_PRINCIPAL:,} - "
        f"Jaccard        : {jaccard_final:.6f}",
        flush=True,
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
        "GENERACIÓN DE SUBMITS ACTIVIDAD",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    candidatos = {
        f"seed{CANONICAL_SEED}":
            ranking_canonical,

        "ensemble5seeds":
            ranking_ensemble,
    }

    submits = []

    for candidato, ranking in candidatos.items():

        print(
            f"\nCandidato: {candidato}",
            flush=True,
        )

        for n in CORTES_ALTERNATIVOS:

            if n > len(ranking):
                raise ValueError(
                    f"N={n:,} supera el score."
                )

            submit = (
                ranking.iloc[
                    :n
                ][
                    [ID_COL]
                ]
                .astype(np.int64)
            )

            submit_path = (
                OUTPUT_DIR
                / (
                    "submit_actividad_"
                    f"{candidato}_"
                    f"n{n}.csv"
                )
            )

            submit.to_csv(
                submit_path,
                index=False,
                header=False,
            )

            validacion = validar_submit(
                path=submit_path,
                ids_score=ids_score,
                n_esperado=n,
            )

            submits.append(
                {
                    "candidato":
                        candidato,

                    "n":
                        n,

                    "path":
                        str(submit_path),

                    "valido":
                        validacion["valido"],
                }
            )

            marca = (
                "  <-- CORTE PRINCIPAL"
                if n == N_PRINCIPAL
                else ""
            )

            print(
                f"  N={n:>5,}: "
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
    # Validación final de ambos candidatos principales
    # ========================================================

    principales = {}

    for candidato, ranking in candidatos.items():

        principal_path = (
            OUTPUT_DIR
            / (
                "submit_actividad_"
                f"{candidato}_"
                f"n{N_PRINCIPAL}.csv"
            )
        )

        principal = pd.read_csv(
            principal_path,
            header=None,
            names=[ID_COL],
        )

        if len(principal) != N_PRINCIPAL:
            raise ValueError(
                f"{candidato}: submit principal "
                f"no tiene {N_PRINCIPAL:,} filas."
            )

        if (
            principal[ID_COL].nunique()
            != N_PRINCIPAL
        ):
            raise ValueError(
                f"{candidato}: submit principal "
                "tiene duplicados."
            )

        principal_set = set(
            principal[ID_COL]
            .astype(np.int64)
            .tolist()
        )

        ranking_set = set(
            ranking.iloc[
                :N_PRINCIPAL
            ][ID_COL]
            .astype(np.int64)
            .tolist()
        )

        if principal_set != ranking_set:
            raise ValueError(
                f"{candidato}: submit principal "
                "no coincide con su ranking."
            )

        principales[
            candidato
        ] = str(
            principal_path
        )

    principal_path = Path(
        principales[
            "ensemble5seeds"
        ]
    )

    principal = pd.read_csv(
        principal_path,
        header=None,
        names=[ID_COL],
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "script":
            "z544_competencia_01_final_actividad.py",

        "dataset":
            str(DATASET),

        "train_months":
            TRAIN_MONTHS,

        "score_month":
            SCORE_MONTH,

        "target":
            "BAJA+2",

        "feature_configuration":
            "FULL457 + ACTIVIDAD_CANALES",

        "n_features_originales":
            len(features_originales),

        "n_features_lag1":
            len(features_lag),

        "n_features_delta_lag1":
            len(features_delta),

        "n_features_lag1_disponible":
            1,

        "n_features_actividad":
            1,

        "n_features_total":
            len(features_modelo),

        "lag1_crudos_usados":
            len(features_lag),

        "lag2_usados":
            0,

        "activity_feature":
            "actividad_canales",

        "transaction_vars":
            TRANSACTION_VARS,

        "n_transaction_vars":
            len(TRANSACTION_VARS),

        "activity_definition":
            "count(transaction_var > 0)",

        "train_rows":
            int(len(train)),

        "train_positivos":
            int(y_train.sum()),

        "score_rows":
            int(len(score)),

        "score_historical_count":
            score_lag_count,

        "score_historical_pct":
            score_lag_pct,

        "seeds":
            SEEDS,

        "canonical_seed":
            CANONICAL_SEED,

        "params":
            PARAMS_BASE,

        "n_principal":
            N_PRINCIPAL,

        "cortes_generados":
            CORTES_ALTERNATIVOS,

        "jaccard_seed_mean":
            float(
                jaccard_df["jaccard"].mean()
            ),

        "jaccard_seed_min":
            float(
                jaccard_df["jaccard"].min()
            ),

        "jaccard_seed_max":
            float(
                jaccard_df["jaccard"].max()
            ),

        "jaccard_ensemble_seed_mean":
            float(
                ensemble_vs_seed_df[
                    "jaccard"
                ].mean()
            ),

        "jaccard_canonical_ensemble":
            float(jaccard_final),

        "canonical_ensemble_intersection":
            int(inter_final),

        "principal_submits":
            principales,
    }

    metadata_path = (
        OUTPUT_DIR
        / "metadata_z544.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # Resumen
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
        "RESUMEN FINAL z544",
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
        f"Train rows         : "
        f"{len(train):,}",
        flush=True,
    )

    print(
        f"Train BAJA+2       : "
        f"{int(y_train.sum()):,}",
        flush=True,
    )

    print(
        f"Score              : "
        f"{SCORE_MONTH}",
        flush=True,
    )

    print(
        f"Score rows         : "
        f"{len(score):,}",
        flush=True,
    )

    print(
        "\nFeatures ACTUAL458:",
        flush=True,
    )

    print(
        f"  originales       : "
        f"{len(features_originales):,}",
        flush=True,
    )

    print(
        f"  lag1 crudos      : "
        f"{len(features_lag):,}",
        flush=True,
    )

    print(
        f"  delta_lag1       : "
        f"{len(features_delta):,}",
        flush=True,
    )

    print(
        "  lag1_disponible  : 1",
        flush=True,
    )

    print(
        "  actividad_canales: 1",
        flush=True,
    )

    print(
        f"  TOTAL            : "
        f"{len(features_modelo):,}",
        flush=True,
    )

    print(
        "  lag2 utilizados  : 0",
        flush=True,
    )

    print(
        f"\nVariables actividad: "
        f"{len(TRANSACTION_VARS)}",
        flush=True,
    )

    print(
        f"Seeds              : "
        f"{len(SEEDS)}",
        flush=True,
    )

    print(
        f"Canonical seed     : "
        f"{CANONICAL_SEED}",
        flush=True,
    )

    print(
        f"Hist. disponible   : "
        f"{score_lag_pct:.4f}%",
        flush=True,
    )

    print(
        f"Jaccard seeds      : "
        f"{jaccard_df['jaccard'].mean():.6f}",
        flush=True,
    )

    print(
        f"Jaccard ens/seeds  : "
        f"{ensemble_vs_seed_df['jaccard'].mean():.6f}",
        flush=True,
    )

    print(
        f"Jaccard canon/ens  : "
        f"{jaccard_final:.6f}",
        flush=True,
    )

    print(
        f"Intersección top "
        f"{N_PRINCIPAL:,}: "
        f"{inter_final:,}",
        flush=True,
    )

    print(
        "\nCANDIDATOS PRINCIPALES:",
        flush=True,
    )

    for candidato, path in principales.items():

        print(
            f"  {candidato}:",
            flush=True,
        )

        print(
            f"    N={N_PRINCIPAL:,}",
            flush=True,
        )

        print(
            f"    {path}",
            flush=True,
        )

    print(
        "\nCortes generados  : "
        f"{CORTES_ALTERNATIVOS}",
        flush=True,
    )

    print(
        f"Metadata           : "
        f"{metadata_path}",
        flush=True,
    )

    print(
        f"Tiempo total       : "
        f"{elapsed_total / 60.0:.2f} min",
        flush=True,
    )

    print(
        "\nZ544 FINALIZADO.",
        flush=True,
    )


if __name__ == "__main__":
    main()