#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
z510_seleccion_delta1.py

DMEyF 2026
Selección incremental de features DELTA1.

Objetivo
--------
Partiendo del resultado de Z509, estudiar si la dirección del cambio
inmediatamente anterior al ancla (delta1) agrega señal predictiva
out-of-time sobre un baseline que ya contiene:

    t0 + Top40 media2

Diseño temporal
---------------
TRAIN = 202104
TEST  = 202105

La selección de features se realiza EXCLUSIVAMENTE con TRAIN.

Experimentos
------------
C0_t0_media2_top40
    t0 + Top40 media2

C1_t0_media2_top40_delta1_top10
    C0 + Top10 delta1

C2_t0_media2_top40_delta1_top20
    C0 + Top20 delta1

C3_t0_media2_top40_delta1_top40
    C0 + Top40 delta1

C4_t0_media2_top40_delta1_top80
    C0 + Top80 delta1

C5_t0_media2_top40_delta1_all
    C0 + todas las delta1 válidas

Importante
----------
1. TEST nunca participa de la selección.
2. El ranking MEDIA2 reproduce el criterio de Z509:
       modelo = t0 + todas las media2
       ranking = gain medio entre semillas, sólo TRAIN.
3. El ranking DELTA1 es condicional:
       modelo = t0 + Top40 media2 + todas las delta1
       ranking = gain medio entre semillas, sólo TRAIN.
4. slope2 no se prueba porque con dos puntos equiespaciados:
       slope2 = delta1
5. Se intenta validar C0 contra B3 de Z509.

Salidas
-------
/data/dmeyf/datasets/evaluacion_clusters/seleccion_delta1/

    columnas_eliminadas_train.csv

    ranking_media2_por_seed.csv
    ranking_media2_train.csv
    media2_top40.csv

    ranking_delta1_por_seed.csv
    ranking_delta1_train.csv

    metricas_corridas.csv
    resumen_modelos.csv
    comparacion_baseline.csv

    predicciones_test.csv
    predicciones_ensemble.csv
    resumen_ensemble.csv

    importancia_gain.csv
    importancia_resumen.csv

    diagnostico_saturacion.csv
    validacion_c0_vs_z509.csv

    resumen_z510.txt
    z510.log              # si se ejecuta redirigiendo stdout/stderr
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb

from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURACIÓN
# ======================================================================

INPUT = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "prebaja_features/features_prebaja.csv"
)

OUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "seleccion_delta1"
)

Z509_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "seleccion_media2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_FOTO = 202104
TEST_FOTO = 202105

SEEDS = [
    100,
    200,
    300,
    400,
    500,
]

TOP_N = [
    10,
    20,
    40,
    80,
]

N_MEDIA_BASE = 40

PARAMS = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbosity": -1,
}


# ======================================================================
# UTILIDADES
# ======================================================================

def titulo(txt: str) -> None:

    print()
    print("=" * 78)
    print(txt)
    print("=" * 78)


def subtitulo(txt: str) -> None:

    print()
    print("-" * 78)
    print(txt)
    print("-" * 78)


def nuevo_modelo(seed: int):

    return lgb.LGBMClassifier(
        **PARAMS,
        random_state=seed,
    )


def safe_auc(y, p):

    try:
        return float(
            roc_auc_score(
                y,
                p,
            )
        )
    except Exception:
        return np.nan


def safe_ap(y, p):

    try:
        return float(
            average_precision_score(
                y,
                p,
            )
        )
    except Exception:
        return np.nan


def safe_logloss(y, p):

    try:
        return float(
            log_loss(
                y,
                p,
                labels=[0, 1],
            )
        )
    except Exception:
        return np.nan


# ======================================================================
# CARGA
# ======================================================================

def cargar_datos() -> pd.DataFrame:

    titulo("CARGA DE FEATURE MATRIX")

    if not INPUT.exists():
        raise FileNotFoundError(
            f"No existe:\n{INPUT}"
        )

    print(INPUT)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    print(
        f"Filas    : {len(df):,}"
    )

    print(
        f"Columnas : {df.shape[1]:,}"
    )

    requeridas = {
        "numero_de_cliente",
        "grupo",
        "foto_ancla",
        "cluster",
    }

    faltantes = (
        requeridas
        - set(df.columns)
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas: "
            f"{sorted(faltantes)}"
        )

    # Target.
    df["target"] = (
        df["grupo"]
        .eq("BAJA+2")
        .astype("int8")
    )

    return df


# ======================================================================
# SPLIT TEMPORAL
# ======================================================================

def construir_split(
    df: pd.DataFrame,
):

    titulo("SPLIT TEMPORAL")

    fotos = sorted(
        df["foto_ancla"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        "Fotos disponibles:",
        fotos,
    )

    if TRAIN_FOTO not in fotos:
        raise ValueError(
            f"No existe TRAIN_FOTO={TRAIN_FOTO}"
        )

    if TEST_FOTO not in fotos:
        raise ValueError(
            f"No existe TEST_FOTO={TEST_FOTO}"
        )

    if TRAIN_FOTO >= TEST_FOTO:
        raise ValueError(
            "TRAIN_FOTO debe ser anterior "
            "a TEST_FOTO."
        )

    train = (
        df[
            df["foto_ancla"]
            == TRAIN_FOTO
        ]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        df[
            df["foto_ancla"]
            == TEST_FOTO
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"TRAIN {TRAIN_FOTO}: "
        f"{len(train):,} filas"
    )

    print(
        f"TEST  {TEST_FOTO}: "
        f"{len(test):,} filas"
    )

    print()

    print("Target TRAIN:")
    print(
        train["grupo"]
        .value_counts()
    )

    print()

    print("Target TEST:")
    print(
        test["grupo"]
        .value_counts()
    )

    print()

    print(
        "Clientes TRAIN:",
        f"{train['numero_de_cliente'].nunique():,}"
    )

    print(
        "Clientes TEST :",
        f"{test['numero_de_cliente'].nunique():,}"
    )

    return train, test


# ======================================================================
# VALIDACIÓN DE COLUMNAS
# ======================================================================

def columnas_validas_train(
    train: pd.DataFrame,
    columnas: list[str],
):

    validas = []
    eliminadas = []

    for col in columnas:

        x = pd.to_numeric(
            train[col],
            errors="coerce",
        )

        n = int(
            x.notna().sum()
        )

        if n == 0:

            eliminadas.append(
                (
                    col,
                    "sin_datos_train",
                )
            )

            continue

        nunique = int(
            x.nunique(
                dropna=True
            )
        )

        if nunique <= 1:

            eliminadas.append(
                (
                    col,
                    "constante_train",
                )
            )

            continue

        validas.append(col)

    return validas, eliminadas


# ======================================================================
# DETECCIÓN DE FAMILIAS
# ======================================================================

def detectar_features(
    train: pd.DataFrame,
):

    titulo("DETECCIÓN DE FEATURES")

    t0_cols = sorted(
        [
            c
            for c in train.columns
            if c.endswith("__t0")
        ]
    )

    media2_cols = sorted(
        [
            c
            for c in train.columns
            if c.endswith("__media2")
        ]
    )

    delta1_cols = sorted(
        [
            c
            for c in train.columns
            if c.endswith("__delta1")
        ]
    )

    print(
        f"t0 declaradas     : "
        f"{len(t0_cols):,}"
    )

    print(
        f"media2 declaradas : "
        f"{len(media2_cols):,}"
    )

    print(
        f"delta1 declaradas : "
        f"{len(delta1_cols):,}"
    )

    t0_validas, t0_elim = (
        columnas_validas_train(
            train,
            t0_cols,
        )
    )

    media_validas, media_elim = (
        columnas_validas_train(
            train,
            media2_cols,
        )
    )

    delta1_validas, delta1_elim = (
        columnas_validas_train(
            train,
            delta1_cols,
        )
    )

    print()

    print(
        f"t0 válidas TRAIN     : "
        f"{len(t0_validas):,}"
    )

    print(
        f"media2 válidas TRAIN : "
        f"{len(media_validas):,}"
    )

    print(
        f"delta1 válidas TRAIN : "
        f"{len(delta1_validas):,}"
    )

    print()

    print(
        f"t0 eliminadas        : "
        f"{len(t0_elim):,}"
    )

    print(
        f"media2 eliminadas    : "
        f"{len(media_elim):,}"
    )

    print(
        f"delta1 eliminadas    : "
        f"{len(delta1_elim):,}"
    )

    eliminadas = []

    for familia, lista in [
        ("t0", t0_elim),
        ("media2", media_elim),
        ("delta1", delta1_elim),
    ]:

        for col, motivo in lista:

            eliminadas.append(
                {
                    "familia": familia,
                    "feature": col,
                    "motivo": motivo,
                }
            )

    pd.DataFrame(
        eliminadas,
        columns=[
            "familia",
            "feature",
            "motivo",
        ],
    ).to_csv(
        OUT_DIR
        / "columnas_eliminadas_train.csv",
        index=False,
    )

    if len(t0_validas) == 0:
        raise ValueError(
            "No hay features t0 válidas."
        )

    if len(media_validas) < N_MEDIA_BASE:
        raise ValueError(
            "No hay suficientes features "
            "media2 para Top40."
        )

    if len(delta1_validas) == 0:
        raise ValueError(
            "No hay features delta1 válidas."
        )

    return (
        t0_validas,
        media_validas,
        delta1_validas,
    )


# ======================================================================
# RANKING GENÉRICO POR GAIN — SÓLO TRAIN
# ======================================================================

def ranking_familia_train(
    train: pd.DataFrame,
    base_cols: list[str],
    familia_cols: list[str],
    sufijo: str,
    nombre: str,
):

    titulo(
        f"RANKING {nombre} — "
        "EXCLUSIVAMENTE TRAIN"
    )

    # Evitamos duplicados por seguridad.
    columnas = list(
        dict.fromkeys(
            list(base_cols)
            + list(familia_cols)
        )
    )

    X_train = (
        train[columnas]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .astype("float32")
    )

    y_train = (
        train["target"]
        .to_numpy(
            dtype="int8"
        )
    )

    registros = []

    for seed in SEEDS:

        print(
            f"seed={seed} ...",
            end=" ",
            flush=True,
        )

        inicio = time.time()

        model = nuevo_modelo(seed)

        model.fit(
            X_train,
            y_train,
        )

        gain = (
            model.booster_
            .feature_importance(
                importance_type="gain"
            )
        )

        total = float(
            gain.sum()
        )

        if total > 0:

            share = (
                gain
                / total
            )

        else:

            share = np.zeros_like(
                gain,
                dtype=float,
            )

        tmp = pd.DataFrame(
            {
                "seed": seed,
                "feature": columnas,
                "gain": gain,
                "gain_share": share,
            }
        )

        tmp = tmp[
            tmp["feature"]
            .str.endswith(sufijo)
        ].copy()

        tmp = tmp.sort_values(
            [
                "gain_share",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(
            drop=True
        )

        tmp["rank_seed"] = (
            np.arange(
                1,
                len(tmp) + 1,
            )
        )

        registros.append(tmp)

        print(
            f"{time.time() - inicio:.2f}s"
        )

    ranking_seed = pd.concat(
        registros,
        ignore_index=True,
    )

    slug = nombre.lower()

    ranking_seed.to_csv(
        OUT_DIR
        / f"ranking_{slug}_por_seed.csv",
        index=False,
    )

    resumen = (
        ranking_seed
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            gain_mean=(
                "gain",
                "mean",
            ),
            gain_std=(
                "gain",
                "std",
            ),
            gain_share_mean=(
                "gain_share",
                "mean",
            ),
            gain_share_std=(
                "gain_share",
                "std",
            ),
            rank_mean=(
                "rank_seed",
                "mean",
            ),
            rank_min=(
                "rank_seed",
                "min",
            ),
            rank_max=(
                "rank_seed",
                "max",
            ),
        )
    )

    for n in TOP_N:

        ids = (
            ranking_seed[
                ranking_seed[
                    "rank_seed"
                ]
                <= n
            ]
            .groupby(
                "feature"
            )
            .size()
        )

        resumen[
            f"veces_top{n}"
        ] = (
            resumen["feature"]
            .map(ids)
            .fillna(0)
            .astype(int)
        )

    resumen = resumen.sort_values(
        [
            "gain_share_mean",
            "rank_mean",
            "feature",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    resumen[
        "rank_consenso"
    ] = (
        np.arange(
            1,
            len(resumen) + 1,
        )
    )

    resumen.to_csv(
        OUT_DIR
        / f"ranking_{slug}_train.csv",
        index=False,
    )

    print()

    print(
        f"TOP 25 {nombre} — ranking TRAIN"
    )

    cols_print = [
        "rank_consenso",
        "feature",
        "gain_share_mean",
        "rank_mean",
        "rank_min",
        "rank_max",
        "veces_top10",
        "veces_top20",
        "veces_top40",
    ]

    print(
        resumen[
            cols_print
        ]
        .head(25)
        .to_string(
            index=False
        )
    )

    return (
        ranking_seed,
        resumen,
    )


# ======================================================================
# RANKING MEDIA2
# ======================================================================

def ranking_media2_train(
    train,
    t0_cols,
    media2_cols,
):

    return ranking_familia_train(
        train=train,
        base_cols=t0_cols,
        familia_cols=media2_cols,
        sufijo="__media2",
        nombre="media2",
    )


# ======================================================================
# TOP40 MEDIA2
# ======================================================================

def seleccionar_media2_top40(
    ranking_media2,
    media2_cols,
):

    titulo(
        "SELECCIÓN BASE — TOP40 MEDIA2"
    )

    permitidas = set(
        media2_cols
    )

    orden = [
        c
        for c in ranking_media2[
            "feature"
        ].tolist()
        if c in permitidas
    ]

    top40 = orden[
        :N_MEDIA_BASE
    ]

    if len(top40) != N_MEDIA_BASE:
        raise ValueError(
            "No fue posible seleccionar "
            "40 features media2."
        )

    pd.DataFrame(
        {
            "rank": np.arange(
                1,
                len(top40) + 1,
            ),
            "feature": top40,
        }
    ).to_csv(
        OUT_DIR
        / "media2_top40.csv",
        index=False,
    )

    print(
        f"Seleccionadas: {len(top40)}"
    )

    print()

    for i, col in enumerate(
        top40,
        start=1,
    ):

        print(
            f"{i:3d}. {col}"
        )

    return top40


# ======================================================================
# RANKING DELTA1 CONDICIONAL A C0
# ======================================================================

def ranking_delta1_train(
    train,
    t0_cols,
    media2_top40,
    delta1_cols,
):

    # La familia delta1 compite contra toda
    # la información ya disponible en C0.
    base_c0 = (
        list(t0_cols)
        + list(media2_top40)
    )

    return ranking_familia_train(
        train=train,
        base_cols=base_c0,
        familia_cols=delta1_cols,
        sufijo="__delta1",
        nombre="delta1",
    )


# ======================================================================
# CONSTRUCCIÓN DE DISEÑOS C0-C5
# ======================================================================

def construir_disenos(
    t0_cols,
    media2_top40,
    delta1_cols,
    ranking_delta1,
):

    titulo(
        "CONSTRUCCIÓN DE EXPERIMENTOS C0-C5"
    )

    permitidas = set(
        delta1_cols
    )

    orden_delta = [
        c
        for c in ranking_delta1[
            "feature"
        ].tolist()
        if c in permitidas
    ]

    base = (
        list(t0_cols)
        + list(media2_top40)
    )

    diseños = {
        "C0_t0_media2_top40":
            list(base),
    }

    for i, n in enumerate(
        TOP_N,
        start=1,
    ):

        diseños[
            f"C{i}_t0_media2_top40"
            f"_delta1_top{n}"
        ] = (
            list(base)
            + orden_delta[:n]
        )

    diseños[
        "C5_t0_media2_top40_delta1_all"
    ] = (
        list(base)
        + list(delta1_cols)
    )

    # Eliminación defensiva de duplicados.
    diseños = {
        nombre:
            list(
                dict.fromkeys(cols)
            )
        for nombre, cols
        in diseños.items()
    }

    for nombre, cols in diseños.items():

        n_t0 = sum(
            c.endswith("__t0")
            for c in cols
        )

        n_media = sum(
            c.endswith("__media2")
            for c in cols
        )

        n_delta = sum(
            c.endswith("__delta1")
            for c in cols
        )

        print(
            f"{nombre:42s} "
            f"{len(cols):4d} features "
            f"(t0={n_t0}, "
            f"media2={n_media}, "
            f"delta1={n_delta})"
        )

    return diseños


# ======================================================================
# ENTRENAMIENTO / EVALUACIÓN
# ======================================================================

def evaluar_disenos(
    train,
    test,
    diseños,
):

    titulo(
        "VALIDACIÓN PREDICTIVA OUT-OF-TIME"
    )

    resultados = []
    predicciones = []
    importancias = []

    y_train = (
        train["target"]
        .to_numpy(
            dtype="int8"
        )
    )

    y_test = (
        test["target"]
        .to_numpy(
            dtype="int8"
        )
    )

    for nombre, columnas in (
        diseños.items()
    ):

        subtitulo(
            f"EXPERIMENTO {nombre}"
        )

        print(
            f"Features: {len(columnas):,}"
        )

        X_train = (
            train[columnas]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .astype("float32")
        )

        X_test = (
            test[columnas]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .astype("float32")
        )

        for seed in SEEDS:

            print(
                f"seed={seed} ...",
                end=" ",
                flush=True,
            )

            inicio = time.time()

            model = nuevo_modelo(
                seed
            )

            model.fit(
                X_train,
                y_train,
            )

            fit_seconds = (
                time.time()
                - inicio
            )

            p_train = (
                model.predict_proba(
                    X_train
                )[:, 1]
            )

            p_test = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )

            auc_train = safe_auc(
                y_train,
                p_train,
            )

            auc_test = safe_auc(
                y_test,
                p_test,
            )

            ap_test = safe_ap(
                y_test,
                p_test,
            )

            ll_test = safe_logloss(
                y_test,
                p_test,
            )

            resultados.append(
                {
                    "experimento": nombre,
                    "seed": seed,
                    "n_features":
                        len(columnas),
                    "auc_train":
                        auc_train,
                    "auc_test":
                        auc_test,
                    "ap_test":
                        ap_test,
                    "logloss_test":
                        ll_test,
                    "fit_seconds":
                        fit_seconds,
                }
            )

            predicciones.append(
                pd.DataFrame(
                    {
                        "experimento":
                            nombre,
                        "seed":
                            seed,
                        "numero_de_cliente":
                            test[
                                "numero_de_cliente"
                            ].to_numpy(),
                        "foto_ancla":
                            TEST_FOTO,
                        "y":
                            y_test,
                        "pred":
                            p_test,
                    }
                )
            )

            gain = (
                model.booster_
                .feature_importance(
                    importance_type="gain"
                )
            )

            total = float(
                gain.sum()
            )

            if total > 0:
                share = gain / total
            else:
                share = np.zeros_like(
                    gain,
                    dtype=float,
                )

            importancias.append(
                pd.DataFrame(
                    {
                        "experimento":
                            nombre,
                        "seed":
                            seed,
                        "feature":
                            columnas,
                        "gain":
                            gain,
                        "gain_share":
                            share,
                    }
                )
            )

            print(
                f"AUC={auc_test:.6f} "
                f"AP={ap_test:.6f} "
                f"LL={ll_test:.6f} "
                f"{fit_seconds:.2f}s"
            )

    metricas = pd.DataFrame(
        resultados
    )

    pred = pd.concat(
        predicciones,
        ignore_index=True,
    )

    imp = pd.concat(
        importancias,
        ignore_index=True,
    )

    return (
        metricas,
        pred,
        imp,
    )


# ======================================================================
# RESUMEN DE MODELOS
# ======================================================================

def resumir_modelos(
    metricas,
):

    titulo(
        "RESUMEN DE MODELOS"
    )

    resumen = (
        metricas
        .groupby(
            "experimento",
            as_index=False,
        )
        .agg(
            n_features=(
                "n_features",
                "first",
            ),
            auc_test_mean=(
                "auc_test",
                "mean",
            ),
            auc_test_std=(
                "auc_test",
                "std",
            ),
            auc_test_min=(
                "auc_test",
                "min",
            ),
            auc_test_max=(
                "auc_test",
                "max",
            ),
            ap_test_mean=(
                "ap_test",
                "mean",
            ),
            ap_test_std=(
                "ap_test",
                "std",
            ),
            logloss_test_mean=(
                "logloss_test",
                "mean",
            ),
            logloss_test_std=(
                "logloss_test",
                "std",
            ),
            fit_seconds_mean=(
                "fit_seconds",
                "mean",
            ),
        )
    )

    orden = {
        "C0_t0_media2_top40": 0,
        "C1_t0_media2_top40_delta1_top10": 1,
        "C2_t0_media2_top40_delta1_top20": 2,
        "C3_t0_media2_top40_delta1_top40": 3,
        "C4_t0_media2_top40_delta1_top80": 4,
        "C5_t0_media2_top40_delta1_all": 5,
    }

    resumen[
        "_orden"
    ] = (
        resumen["experimento"]
        .map(orden)
        .fillna(999)
    )

    resumen = (
        resumen
        .sort_values("_orden")
        .drop(
            columns="_orden"
        )
        .reset_index(
            drop=True
        )
    )

    base = (
        resumen[
            resumen["experimento"]
            == "C0_t0_media2_top40"
        ]
    )

    if len(base) != 1:
        raise ValueError(
            "No se encontró C0."
        )

    auc_base = float(
        base[
            "auc_test_mean"
        ].iloc[0]
    )

    ap_base = float(
        base[
            "ap_test_mean"
        ].iloc[0]
    )

    resumen[
        "delta_auc_mean"
    ] = (
        resumen[
            "auc_test_mean"
        ]
        - auc_base
    )

    resumen[
        "delta_ap_mean"
    ] = (
        resumen[
            "ap_test_mean"
        ]
        - ap_base
    )

    # Número de semillas en las que
    # cada diseño supera C0.
    base_seed = (
        metricas[
            metricas["experimento"]
            == "C0_t0_media2_top40"
        ][
            [
                "seed",
                "auc_test",
            ]
        ]
        .rename(
            columns={
                "auc_test":
                    "auc_base"
            }
        )
    )

    comp = metricas.merge(
        base_seed,
        on="seed",
        how="left",
    )

    comp[
        "mejora_auc"
    ] = (
        comp["auc_test"]
        > comp["auc_base"]
    )

    wins = (
        comp
        .groupby(
            "experimento"
        )[
            "mejora_auc"
        ]
        .sum()
    )

    resumen[
        "semillas_mejor_auc"
    ] = (
        resumen["experimento"]
        .map(wins)
        .fillna(0)
        .astype(int)
    )

    print(
        resumen.to_string(
            index=False
        )
    )

    return resumen


# ======================================================================
# COMPARACIÓN CORRIDA A CORRIDA CONTRA C0
# ======================================================================

def comparar_baseline(
    metricas,
):

    base = (
        metricas[
            metricas["experimento"]
            == "C0_t0_media2_top40"
        ][
            [
                "seed",
                "auc_test",
                "ap_test",
                "logloss_test",
            ]
        ]
        .rename(
            columns={
                "auc_test":
                    "auc_c0",
                "ap_test":
                    "ap_c0",
                "logloss_test":
                    "logloss_c0",
            }
        )
    )

    x = metricas.merge(
        base,
        on="seed",
        how="left",
    )

    x[
        "delta_auc_vs_c0"
    ] = (
        x["auc_test"]
        - x["auc_c0"]
    )

    x[
        "delta_ap_vs_c0"
    ] = (
        x["ap_test"]
        - x["ap_c0"]
    )

    x[
        "delta_logloss_vs_c0"
    ] = (
        x["logloss_test"]
        - x["logloss_c0"]
    )

    return x


# ======================================================================
# ENSEMBLE ENTRE SEMILLAS
# ======================================================================

def construir_ensemble(
    pred,
):

    titulo(
        "ENSEMBLE ENTRE SEMILLAS"
    )

    ens = (
        pred
        .groupby(
            [
                "experimento",
                "numero_de_cliente",
                "foto_ancla",
                "y",
            ],
            as_index=False,
        )
        .agg(
            pred_mean=(
                "pred",
                "mean",
            ),
            pred_std=(
                "pred",
                "std",
            ),
        )
    )

    filas = []

    for nombre, g in (
        ens.groupby(
            "experimento",
            sort=False,
        )
    ):

        y = g[
            "y"
        ].to_numpy()

        p = g[
            "pred_mean"
        ].to_numpy()

        filas.append(
            {
                "experimento":
                    nombre,
                "n":
                    len(g),
                "auc_ensemble":
                    safe_auc(y, p),
                "ap_ensemble":
                    safe_ap(y, p),
                "logloss_ensemble":
                    safe_logloss(y, p),
            }
        )

    resumen = pd.DataFrame(
        filas
    )

    print(
        resumen.to_string(
            index=False
        )
    )

    return (
        ens,
        resumen,
    )


# ======================================================================
# IMPORTANCIA
# ======================================================================

def resumir_importancia(
    imp,
):

    resumen = (
        imp
        .groupby(
            [
                "experimento",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            gain_mean=(
                "gain",
                "mean",
            ),
            gain_std=(
                "gain",
                "std",
            ),
            gain_share_mean=(
                "gain_share",
                "mean",
            ),
            gain_share_std=(
                "gain_share",
                "std",
            ),
        )
    )

    resumen[
        "rank"
    ] = (
        resumen
        .groupby(
            "experimento"
        )[
            "gain_share_mean"
        ]
        .rank(
            method="first",
            ascending=False,
        )
    )

    return (
        resumen
        .sort_values(
            [
                "experimento",
                "rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ======================================================================
# VALIDACIÓN C0 CONTRA Z509
# ======================================================================

def validar_c0_z509(
    metricas,
    resumen,
):

    titulo(
        "VALIDACIÓN C0 CONTRA Z509"
    )

    path_metricas = (
        Z509_DIR
        / "metricas_corridas.csv"
    )

    path_resumen = (
        Z509_DIR
        / "resumen_modelos.csv"
    )

    registros = []

    c0 = (
        metricas[
            metricas["experimento"]
            == "C0_t0_media2_top40"
        ][
            [
                "seed",
                "auc_test",
                "ap_test",
                "logloss_test",
            ]
        ]
        .copy()
    )

    if path_metricas.exists():

        z = pd.read_csv(
            path_metricas
        )

        b3 = z[
            z["experimento"]
            == "B3_t0_media2_top40"
        ].copy()

        if not b3.empty:

            comp = c0.merge(
                b3[
                    [
                        "seed",
                        "auc_test",
                        "ap_test",
                        "logloss_test",
                    ]
                ],
                on="seed",
                how="inner",
                suffixes=(
                    "_c0",
                    "_b3",
                ),
            )

            comp[
                "delta_auc"
            ] = (
                comp["auc_test_c0"]
                - comp["auc_test_b3"]
            )

            comp[
                "delta_ap"
            ] = (
                comp["ap_test_c0"]
                - comp["ap_test_b3"]
            )

            comp[
                "delta_logloss"
            ] = (
                comp[
                    "logloss_test_c0"
                ]
                - comp[
                    "logloss_test_b3"
                ]
            )

            comp.to_csv(
                OUT_DIR
                / "validacion_c0_vs_z509.csv",
                index=False,
            )

            print(
                comp.to_string(
                    index=False
                )
            )

            max_abs = float(
                comp[
                    "delta_auc"
                ]
                .abs()
                .max()
            )

            print()

            print(
                "Máxima diferencia absoluta "
                f"de AUC: {max_abs:.12f}"
            )

            if max_abs < 1e-10:

                print(
                    "CONTROL C0 vs B3: "
                    "REPRODUCCIÓN EXACTA."
                )

            elif max_abs < 1e-4:

                print(
                    "CONTROL C0 vs B3: "
                    "diferencia mínima."
                )

            else:

                print(
                    "ADVERTENCIA: C0 no reproduce "
                    "exactamente B3."
                )

            return comp

    # Fallback con resumen.
    if path_resumen.exists():

        z = pd.read_csv(
            path_resumen
        )

        b3 = z[
            z["experimento"]
            == "B3_t0_media2_top40"
        ]

        c0r = resumen[
            resumen["experimento"]
            == "C0_t0_media2_top40"
        ]

        if (
            not b3.empty
            and not c0r.empty
        ):

            auc_b3 = float(
                b3[
                    "auc_test_mean"
                ].iloc[0]
            )

            auc_c0 = float(
                c0r[
                    "auc_test_mean"
                ].iloc[0]
            )

            print(
                f"B3 Z509: {auc_b3:.9f}"
            )

            print(
                f"C0 Z510: {auc_c0:.9f}"
            )

            print(
                "Delta   : "
                f"{auc_c0 - auc_b3:+.9f}"
            )

    print(
        "No fue posible realizar la "
        "comparación corrida a corrida."
    )

    return pd.DataFrame(
        registros
    )


# ======================================================================
# DIAGNÓSTICO DE SATURACIÓN
# ======================================================================

def diagnostico_saturacion(
    resumen,
):

    cols = [
        "experimento",
        "n_features",
        "auc_test_mean",
        "auc_test_std",
        "ap_test_mean",
        "logloss_test_mean",
        "delta_auc_mean",
        "delta_ap_mean",
        "semillas_mejor_auc",
    ]

    x = resumen[
        cols
    ].copy()

    titulo(
        "DIAGNÓSTICO DE SATURACIÓN"
    )

    print(
        x.to_string(
            index=False
        )
    )

    mejor = (
        x.sort_values(
            "auc_test_mean",
            ascending=False,
        )
        .iloc[0]
    )

    print()

    print(
        "Mayor AUC medio:"
    )

    print(
        f"  {mejor['experimento']}"
    )

    print(
        "  AUC = "
        f"{mejor['auc_test_mean']:.6f}"
    )

    print(
        "  features = "
        f"{int(mejor['n_features'])}"
    )

    return x


# ======================================================================
# RESUMEN TXT
# ======================================================================

def escribir_resumen(
    resumen,
    ensemble,
    media2_top40,
    ranking_delta1,
):

    path = (
        OUT_DIR
        / "resumen_z510.txt"
    )

    mejor_auc = (
        resumen
        .sort_values(
            "auc_test_mean",
            ascending=False,
        )
        .iloc[0]
    )

    mejor_ap = (
        resumen
        .sort_values(
            "ap_test_mean",
            ascending=False,
        )
        .iloc[0]
    )

    top_delta = (
        ranking_delta1[
            [
                "rank_consenso",
                "feature",
                "gain_share_mean",
                "rank_mean",
            ]
        ]
        .head(25)
        .to_string(
            index=False
        )
    )

    tabla = resumen.to_string(
        index=False
    )

    tabla_ens = ensemble.to_string(
        index=False
    )

    texto = f"""
Z510 — SELECCIÓN INCREMENTAL DELTA1
===================================

Objetivo
--------
Evaluar si delta1 agrega información predictiva out-of-time
sobre el baseline:

    t0 + Top40 media2

TRAIN = {TRAIN_FOTO}
TEST  = {TEST_FOTO}

Semillas
--------
{SEEDS}

Baseline
--------
C0_t0_media2_top40

Cantidad de media2 fijadas:
{len(media2_top40)}

Metodología
-----------
El ranking de media2 se construye exclusivamente con TRAIN.

Luego se fija Top40 media2.

El ranking de delta1 se construye exclusivamente con TRAIN,
condicionado a:

    t0 + Top40 media2

TEST no participa de ninguna selección.

Resultados
----------
{tabla}

Ensemble
--------
{tabla_ens}

Mayor AUC medio
---------------
Experimento:
{mejor_auc['experimento']}

AUC:
{mejor_auc['auc_test_mean']:.9f}

Mayor AP medio
--------------
Experimento:
{mejor_ap['experimento']}

AP:
{mejor_ap['ap_test_mean']:.9f}

Top 25 delta1 en TRAIN
----------------------
{top_delta}

Interpretación
--------------
La comparación relevante es incremental respecto de C0.

No debe interpretarse el ranking de gain como causalidad.

La decisión sobre incorporar delta1 debe considerar conjuntamente:

- AUC out-of-time;
- AP out-of-time;
- logloss;
- estabilidad entre semillas;
- número de features;
- magnitud de la mejora respecto de C0.
"""

    path.write_text(
        texto.strip()
        + "\n",
        encoding="utf-8",
    )


# ======================================================================
# EXPORTACIÓN
# ======================================================================

def exportar(
    metricas,
    resumen,
    comparacion,
    pred,
    ens,
    resumen_ens,
    imp,
    imp_resumen,
    diagnostico,
):

    titulo(
        "EXPORTACIÓN"
    )

    archivos = {
        "metricas_corridas.csv":
            metricas,

        "resumen_modelos.csv":
            resumen,

        "comparacion_baseline.csv":
            comparacion,

        "predicciones_test.csv":
            pred,

        "predicciones_ensemble.csv":
            ens,

        "resumen_ensemble.csv":
            resumen_ens,

        "importancia_gain.csv":
            imp,

        "importancia_resumen.csv":
            imp_resumen,

        "diagnostico_saturacion.csv":
            diagnostico,
    }

    for nombre, df in archivos.items():

        path = (
            OUT_DIR
            / nombre
        )

        df.to_csv(
            path,
            index=False,
        )

        print(
            f"OK  {nombre:35s} "
            f"{path.stat().st_size:12,} bytes"
        )


# ======================================================================
# MAIN
# ======================================================================

def main():

    inicio_total = time.time()

    titulo(
        "Z510 — SELECCIÓN INCREMENTAL DELTA1"
    )

    print(
        """
Pregunta experimental:

¿La dirección del cambio reciente (delta1)
agrega señal predictiva out-of-time después de
controlar por:

    t0 + Top40 media2

Toda selección se realiza exclusivamente en TRAIN.
"""
    )

    # --------------------------------------------------------------
    # Datos
    # --------------------------------------------------------------

    df = cargar_datos()

    train, test = construir_split(
        df
    )

    # --------------------------------------------------------------
    # Familias
    # --------------------------------------------------------------

    (
        t0_cols,
        media2_cols,
        delta1_cols,
    ) = detectar_features(
        train
    )

    # --------------------------------------------------------------
    # Ranking MEDIA2
    #
    # Reproduce el criterio de Z509.
    # --------------------------------------------------------------

    (
        ranking_media_seed,
        ranking_media,
    ) = ranking_media2_train(
        train,
        t0_cols,
        media2_cols,
    )

    # --------------------------------------------------------------
    # Fijamos Top40 MEDIA2
    # --------------------------------------------------------------

    media2_top40 = (
        seleccionar_media2_top40(
            ranking_media,
            media2_cols,
        )
    )

    # --------------------------------------------------------------
    # Ranking DELTA1 condicionado a C0
    # --------------------------------------------------------------

    (
        ranking_delta_seed,
        ranking_delta,
    ) = ranking_delta1_train(
        train,
        t0_cols,
        media2_top40,
        delta1_cols,
    )

    # --------------------------------------------------------------
    # Diseños
    # --------------------------------------------------------------

    diseños = construir_disenos(
        t0_cols,
        media2_top40,
        delta1_cols,
        ranking_delta,
    )

    # --------------------------------------------------------------
    # Evaluación
    # --------------------------------------------------------------

    (
        metricas,
        pred,
        imp,
    ) = evaluar_disenos(
        train,
        test,
        diseños,
    )

    # --------------------------------------------------------------
    # Resúmenes
    # --------------------------------------------------------------

    resumen = resumir_modelos(
        metricas
    )

    comparacion = comparar_baseline(
        metricas
    )

    (
        ens,
        resumen_ens,
    ) = construir_ensemble(
        pred
    )

    imp_resumen = (
        resumir_importancia(
            imp
        )
    )

    diagnostico = (
        diagnostico_saturacion(
            resumen
        )
    )

    # --------------------------------------------------------------
    # Control contra Z509
    # --------------------------------------------------------------

    validar_c0_z509(
        metricas,
        resumen,
    )

    # --------------------------------------------------------------
    # Exportación
    # --------------------------------------------------------------

    exportar(
        metricas,
        resumen,
        comparacion,
        pred,
        ens,
        resumen_ens,
        imp,
        imp_resumen,
        diagnostico,
    )

    escribir_resumen(
        resumen,
        resumen_ens,
        media2_top40,
        ranking_delta,
    )

    titulo(
        "RESULTADO FINAL Z510"
    )

    cols_final = [
        "experimento",
        "n_features",
        "auc_test_mean",
        "auc_test_std",
        "ap_test_mean",
        "logloss_test_mean",
        "delta_auc_mean",
        "delta_ap_mean",
        "semillas_mejor_auc",
    ]

    print(
        resumen[
            cols_final
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        "Tiempo total: "
        f"{time.time() - inicio_total:.2f} s"
    )

    print()

    print(
        "Z510 FINALIZADO CORRECTAMENTE."
    )

    print()

    print(
        "Resultados:"
    )

    print(
        OUT_DIR
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        titulo(
            "ERROR EN Z510"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        raise