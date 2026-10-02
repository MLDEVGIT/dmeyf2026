#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
z508_validacion_predictiva.py

DMEyF 2026
Validación predictiva temporal de features pre-BAJA+2.

Objetivo
--------
Evaluar si incorporar historia reciente del cliente agrega capacidad
predictiva respecto de utilizar solamente la observación actual.

Diseño temporal
---------------
TRAIN:
    foto_ancla = 202104

TEST:
    foto_ancla = 202105

Target:
    BAJA+2 = 1
    FIEL   = 0

Experimentos
------------
A0_t0
    Sólo variables en t=0.

A1_t0_delta1
    t=0 + cambio respecto de t=-1.

A2_t0_media2
    t=0 + media de t=-1,0.

A3_t0_delta1_media2
    t=0 + delta1 + media2.

IMPORTANTE
----------
- No se utiliza cluster como predictor.
- No se utilizan features de tres observaciones.
- No se utiliza slope2 porque es algebraicamente equivalente a delta1.
- No se realiza split aleatorio.
- No se tunean hiperparámetros.
- No se optimiza threshold sobre TEST.
- La selección de columnas constantes se realiza exclusivamente en TRAIN.
- Los NaN son tratados nativamente por LightGBM.
- Se utilizan varias semillas para medir estabilidad.

Este experimento NO estima todavía la ganancia económica de competencia.
Su objetivo es determinar si las features temporales mejoran el ranking
predictivo fuera de muestra temporal.

Entradas
--------
/data/dmeyf/datasets/evaluacion_clusters/prebaja_features/
    features_prebaja.csv

Salidas
-------
/data/dmeyf/datasets/evaluacion_clusters/validacion_predictiva/

    metricas_corridas.csv
    resumen_modelos.csv
    comparacion_baseline.csv
    predicciones_test.csv
    importancia_gain.csv
    importancia_resumen.csv
    columnas_modelos.csv
    resumen_z508.txt
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


# =====================================================================
# CONFIGURACIÓN
# =====================================================================

INPUT_FILE = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "prebaja_features/features_prebaja.csv"
)

OUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "validacion_predictiva"
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

META_COLS = [
    "numero_de_cliente",
    "grupo",
    "foto_ancla",
    "cluster",
    "n_historia",
]

TARGET_COL = "target"

TRANSFORMACIONES_PERMITIDAS = {
    "A0_t0": [
        "__t0",
    ],
    "A1_t0_delta1": [
        "__t0",
        "__delta1",
    ],
    "A2_t0_media2": [
        "__t0",
        "__media2",
    ],
    "A3_t0_delta1_media2": [
        "__t0",
        "__delta1",
        "__media2",
    ],
}

TRANSFORMACIONES_PROHIBIDAS = [
    "__delta2",
    "__slope2",
    "__media3",
    "__min3",
    "__max3",
    "__std3",
    "__slope3",
]

LGB_PARAMS = {
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


# =====================================================================
# UTILIDADES
# =====================================================================

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


def detectar_transformacion(col: str) -> str | None:
    """
    Devuelve el sufijo de transformación reconocido.
    """

    sufijos = [
        "__delta1",
        "__delta2",
        "__media2",
        "__slope2",
        "__media3",
        "__min3",
        "__max3",
        "__std3",
        "__slope3",
        "__t0",
    ]

    for suf in sufijos:
        if col.endswith(suf):
            return suf

    return None


def variable_original(col: str) -> str:
    """
    Quita el sufijo de transformación.
    """

    transf = detectar_transformacion(col)

    if transf is None:
        return col

    return col[: -len(transf)]


def safe_auc(
    y_true: pd.Series | np.ndarray,
    prob: np.ndarray,
) -> float:

    try:
        return float(
            roc_auc_score(
                y_true,
                prob,
            )
        )
    except Exception:
        return np.nan


def safe_ap(
    y_true: pd.Series | np.ndarray,
    prob: np.ndarray,
) -> float:

    try:
        return float(
            average_precision_score(
                y_true,
                prob,
            )
        )
    except Exception:
        return np.nan


def safe_logloss(
    y_true: pd.Series | np.ndarray,
    prob: np.ndarray,
) -> float:

    try:
        return float(
            log_loss(
                y_true,
                prob,
                labels=[0, 1],
            )
        )
    except Exception:
        return np.nan


# =====================================================================
# CARGA
# =====================================================================

def cargar_features() -> pd.DataFrame:

    titulo("CARGA DE FEATURE MATRIX")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No existe:\n{INPUT_FILE}"
        )

    print(INPUT_FILE)

    t0 = time.time()

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    elapsed = time.time() - t0

    print(
        f"Dimensiones : "
        f"{df.shape[0]:,} × {df.shape[1]:,}"
    )

    print(
        f"Tiempo carga: "
        f"{elapsed:.2f} s"
    )

    return df


# =====================================================================
# VALIDACIONES INICIALES
# =====================================================================

def validar_estructura(
    df: pd.DataFrame,
) -> None:

    titulo("VALIDACIÓN DE ESTRUCTURA")

    faltantes = [
        c
        for c in META_COLS
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas obligatorias: "
            + ", ".join(faltantes)
        )

    grupos = set(
        df["grupo"]
        .dropna()
        .unique()
    )

    esperados = {
        "BAJA+2",
        "FIEL",
    }

    if grupos != esperados:
        raise ValueError(
            f"Grupos inesperados: {grupos}"
        )

    fotos = sorted(
        df["foto_ancla"]
        .dropna()
        .unique()
        .tolist()
    )

    print("Fotos disponibles:")
    print(fotos)

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
            "TRAIN_FOTO debe ser anterior a TEST_FOTO."
        )

    print()
    print("Distribución por foto y grupo:")

    print(
        pd.crosstab(
            df["foto_ancla"],
            df["grupo"],
            margins=True,
        )
    )

    print()
    print("Estructura: OK")


# =====================================================================
# TARGET
# =====================================================================

def construir_target(
    df: pd.DataFrame,
) -> pd.DataFrame:

    titulo("CONSTRUCCIÓN DEL TARGET")

    df = df.copy()

    df[TARGET_COL] = (
        df["grupo"] == "BAJA+2"
    ).astype("int8")

    print(
        df.groupby(
            [
                "foto_ancla",
                "grupo",
                TARGET_COL,
            ]
        )
        .size()
    )

    if df[TARGET_COL].isna().any():
        raise ValueError(
            "Target contiene NA."
        )

    return df


# =====================================================================
# TRAIN / TEST
# =====================================================================

def construir_train_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    titulo("PARTICIÓN TEMPORAL")

    train = df[
        df["foto_ancla"] == TRAIN_FOTO
    ].copy()

    test = df[
        df["foto_ancla"] == TEST_FOTO
    ].copy()

    if train.empty:
        raise ValueError(
            "TRAIN vacío."
        )

    if test.empty:
        raise ValueError(
            "TEST vacío."
        )

    if train["foto_ancla"].nunique() != 1:
        raise ValueError(
            "TRAIN contiene más de una foto."
        )

    if test["foto_ancla"].nunique() != 1:
        raise ValueError(
            "TEST contiene más de una foto."
        )

    if (
        int(train["foto_ancla"].iloc[0])
        != TRAIN_FOTO
    ):
        raise ValueError(
            "Foto incorrecta en TRAIN."
        )

    if (
        int(test["foto_ancla"].iloc[0])
        != TEST_FOTO
    ):
        raise ValueError(
            "Foto incorrecta en TEST."
        )

    print(
        f"TRAIN {TRAIN_FOTO}: "
        f"{len(train):,} filas"
    )

    print(
        train["grupo"]
        .value_counts()
    )

    print()

    print(
        f"TEST {TEST_FOTO}: "
        f"{len(test):,} filas"
    )

    print(
        test["grupo"]
        .value_counts()
    )

    print()

    ids_train = set(
        train["numero_de_cliente"]
    )

    ids_test = set(
        test["numero_de_cliente"]
    )

    overlap = (
        ids_train
        & ids_test
    )

    print(
        "Clientes TRAIN       :",
        f"{len(ids_train):,}",
    )

    print(
        "Clientes TEST        :",
        f"{len(ids_test):,}",
    )

    print(
        "Clientes compartidos :",
        f"{len(overlap):,}",
    )

    print()
    print(
        "NOTA: la superposición de clientes fieles "
        "es esperada en un diseño cliente-mes."
    )

    return train, test


# =====================================================================
# DEFINICIÓN DE FEATURES
# =====================================================================

def columnas_por_sufijos(
    df: pd.DataFrame,
    sufijos: list[str],
) -> list[str]:

    cols = []

    for col in df.columns:

        if col in META_COLS:
            continue

        if col == TARGET_COL:
            continue

        if any(
            col.endswith(s)
            for s in sufijos
        ):
            cols.append(col)

    return sorted(cols)


def construir_diseno_features(
    df: pd.DataFrame,
) -> dict[str, list[str]]:

    titulo("DEFINICIÓN DE EXPERIMENTOS")

    diseños = {}

    for nombre, sufijos in (
        TRANSFORMACIONES_PERMITIDAS.items()
    ):

        cols = columnas_por_sufijos(
            df,
            sufijos,
        )

        if not cols:
            raise ValueError(
                f"{nombre} no tiene columnas."
            )

        diseños[nombre] = cols

        print(
            f"{nombre:24s}: "
            f"{len(cols):4d} columnas"
        )

    return diseños


# =====================================================================
# VALIDACIÓN DE LOS DISEÑOS
# =====================================================================

def validar_disenos(
    diseños: dict[str, list[str]],
) -> None:

    titulo("VALIDACIÓN DE DISEÑOS")

    # -------------------------------------------------------------
    # Cantidad t0
    # -------------------------------------------------------------

    a0 = set(
        diseños["A0_t0"]
    )

    if not a0:
        raise ValueError(
            "A0 no contiene features t0."
        )

    # -------------------------------------------------------------
    # A0 debe estar incluido exactamente en todos.
    # -------------------------------------------------------------

    for nombre, cols in diseños.items():

        cset = set(cols)

        if not a0.issubset(cset):
            faltan = sorted(
                a0 - cset
            )

            raise ValueError(
                f"{nombre} no contiene todas "
                f"las features A0. "
                f"Faltan {len(faltan)}."
            )

    # -------------------------------------------------------------
    # Ninguna feature prohibida.
    # -------------------------------------------------------------

    for nombre, cols in diseños.items():

        prohibidas = [
            c
            for c in cols
            if any(
                c.endswith(s)
                for s
                in TRANSFORMACIONES_PROHIBIDAS
            )
        ]

        if prohibidas:
            raise ValueError(
                f"{nombre} contiene features "
                f"prohibidas:\n"
                + "\n".join(
                    prohibidas[:20]
                )
            )

    # -------------------------------------------------------------
    # Cluster jamás entra.
    # -------------------------------------------------------------

    for nombre, cols in diseños.items():

        if "cluster" in cols:
            raise ValueError(
                f"cluster entró en {nombre}."
            )

    # -------------------------------------------------------------
    # Validamos transformaciones exactas.
    # -------------------------------------------------------------

    for nombre, cols in diseños.items():

        permitidos = set(
            TRANSFORMACIONES_PERMITIDAS[
                nombre
            ]
        )

        encontrados = set()

        for col in cols:

            t = detectar_transformacion(
                col
            )

            if t is not None:
                encontrados.add(t)

        if encontrados != permitidos:
            raise ValueError(
                f"{nombre}: transformaciones "
                f"inesperadas.\n"
                f"Esperadas: {permitidos}\n"
                f"Encontradas: {encontrados}"
            )

    print(
        "Todos los diseños contienen "
        "exactamente las familias esperadas."
    )

    print(
        "cluster excluido: OK"
    )

    print(
        "features de tres meses excluidas: OK"
    )

    print(
        "slope2 excluido: OK"
    )

    print(
        "A0 incluido en A1/A2/A3: OK"
    )


# =====================================================================
# PREPARACIÓN DE MATRICES
# =====================================================================

def preparar_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columnas: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
]:

    """
    Elimina columnas inútiles mirando EXCLUSIVAMENTE TRAIN.

    Se eliminan:
    - columnas totalmente NA en train;
    - columnas con <= 1 valor distinto no-NA en train.

    La misma selección se aplica luego a TEST.
    """

    X_train = train[
        columnas
    ].copy()

    X_test = test[
        columnas
    ].copy()

    # Conversión numérica explícita.
    for col in columnas:

        X_train[col] = pd.to_numeric(
            X_train[col],
            errors="coerce",
        )

        X_test[col] = pd.to_numeric(
            X_test[col],
            errors="coerce",
        )

    eliminar = []

    for col in columnas:

        s = X_train[col]

        if s.notna().sum() == 0:
            eliminar.append(col)
            continue

        if s.nunique(
            dropna=True
        ) <= 1:
            eliminar.append(col)

    usar = [
        c
        for c in columnas
        if c not in eliminar
    ]

    if not usar:
        raise ValueError(
            "No quedaron columnas utilizables."
        )

    X_train = X_train[
        usar
    ]

    X_test = X_test[
        usar
    ]

    return (
        X_train,
        X_test,
        usar,
        eliminar,
    )


# =====================================================================
# ENTRENAMIENTO
# =====================================================================

def entrenar_experimento(
    nombre: str,
    columnas: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    list[dict],
    list[pd.DataFrame],
    list[pd.DataFrame],
    dict,
]:

    subtitulo(
        f"EXPERIMENTO {nombre}"
    )

    (
        X_train,
        X_test,
        columnas_usadas,
        columnas_eliminadas,
    ) = preparar_matrices(
        train,
        test,
        columnas,
    )

    y_train = (
        train[TARGET_COL]
        .astype("int8")
    )

    y_test = (
        test[TARGET_COL]
        .astype("int8")
    )

    print(
        "Columnas declaradas :",
        f"{len(columnas):,}",
    )

    print(
        "Columnas utilizadas :",
        f"{len(columnas_usadas):,}",
    )

    print(
        "Eliminadas por TRAIN:",
        f"{len(columnas_eliminadas):,}",
    )

    print(
        "TRAIN:",
        X_train.shape,
    )

    print(
        "TEST :",
        X_test.shape,
    )

    metricas = []
    predicciones = []
    importancias = []

    for seed in SEEDS:

        print(
            f"  seed={seed} ... ",
            end="",
            flush=True,
        )

        params = dict(
            LGB_PARAMS
        )

        params["random_state"] = seed

        # LightGBM sólo usa subsample si
        # subsample_freq > 0.
        params["subsample_freq"] = 1

        model = LGBMClassifier(
            **params
        )

        inicio = time.time()

        model.fit(
            X_train,
            y_train,
        )

        segundos = (
            time.time()
            - inicio
        )

        p_train = model.predict_proba(
            X_train
        )[:, 1]

        p_test = model.predict_proba(
            X_test
        )[:, 1]

        auc_train = safe_auc(
            y_train,
            p_train,
        )

        auc_test = safe_auc(
            y_test,
            p_test,
        )

        ap_train = safe_ap(
            y_train,
            p_train,
        )

        ap_test = safe_ap(
            y_test,
            p_test,
        )

        ll_train = safe_logloss(
            y_train,
            p_train,
        )

        ll_test = safe_logloss(
            y_test,
            p_test,
        )

        metricas.append(
            {
                "experimento": nombre,
                "seed": seed,
                "foto_train": TRAIN_FOTO,
                "foto_test": TEST_FOTO,
                "n_train": len(train),
                "n_test": len(test),
                "n_features_declaradas":
                    len(columnas),
                "n_features_usadas":
                    len(columnas_usadas),
                "n_features_eliminadas":
                    len(columnas_eliminadas),
                "auc_train": auc_train,
                "auc_test": auc_test,
                "auc_gap":
                    auc_train - auc_test,
                "ap_train": ap_train,
                "ap_test": ap_test,
                "ap_gap":
                    ap_train - ap_test,
                "logloss_train":
                    ll_train,
                "logloss_test":
                    ll_test,
                "segundos_fit":
                    segundos,
            }
        )

        pred = pd.DataFrame(
            {
                "numero_de_cliente":
                    test[
                        "numero_de_cliente"
                    ].to_numpy(),

                "foto_ancla":
                    test[
                        "foto_ancla"
                    ].to_numpy(),

                "grupo":
                    test[
                        "grupo"
                    ].to_numpy(),

                "target":
                    y_test.to_numpy(),

                "experimento":
                    nombre,

                "seed":
                    seed,

                "prob":
                    p_test,
            }
        )

        predicciones.append(
            pred
        )

        booster = (
            model.booster_
        )

        gain = (
            booster.feature_importance(
                importance_type="gain"
            )
        )

        split = (
            booster.feature_importance(
                importance_type="split"
            )
        )

        imp = pd.DataFrame(
            {
                "experimento":
                    nombre,

                "seed":
                    seed,

                "feature":
                    columnas_usadas,

                "gain":
                    gain,

                "split":
                    split,
            }
        )

        total_gain = (
            imp["gain"]
            .sum()
        )

        if total_gain > 0:

            imp["gain_share"] = (
                imp["gain"]
                / total_gain
            )

        else:

            imp["gain_share"] = (
                np.nan
            )

        imp["variable"] = (
            imp["feature"]
            .map(
                variable_original
            )
        )

        imp["transformacion"] = (
            imp["feature"]
            .map(
                detectar_transformacion
            )
        )

        importancias.append(
            imp
        )

        print(
            f"AUC={auc_test:.6f}  "
            f"AP={ap_test:.6f}  "
            f"LL={ll_test:.6f}  "
            f"{segundos:.2f}s"
        )

    info_columnas = {
        "experimento": nombre,
        "columnas_declaradas":
            columnas,
        "columnas_usadas":
            columnas_usadas,
        "columnas_eliminadas":
            columnas_eliminadas,
    }

    return (
        metricas,
        predicciones,
        importancias,
        info_columnas,
    )


# =====================================================================
# RESUMEN DE MODELOS
# =====================================================================

def resumir_metricas(
    metricas: pd.DataFrame,
) -> pd.DataFrame:

    titulo("RESUMEN DE MODELOS")

    resumen = (
        metricas
        .groupby(
            "experimento",
            as_index=False,
        )
        .agg(
            seeds=(
                "seed",
                "nunique",
            ),

            n_features=(
                "n_features_usadas",
                "first",
            ),

            auc_train_mean=(
                "auc_train",
                "mean",
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
                "segundos_fit",
                "mean",
            ),
        )
    )

    orden = {
        "A0_t0": 0,
        "A1_t0_delta1": 1,
        "A2_t0_media2": 2,
        "A3_t0_delta1_media2": 3,
    }

    resumen["_orden"] = (
        resumen["experimento"]
        .map(orden)
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

    print(
        resumen.to_string(
            index=False
        )
    )

    return resumen


# =====================================================================
# COMPARACIÓN CONTRA BASELINE
# =====================================================================

def comparar_baseline(
    metricas: pd.DataFrame,
) -> pd.DataFrame:

    titulo("COMPARACIÓN CONTRA A0")

    base = (
        metricas[
            metricas["experimento"]
            == "A0_t0"
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
                    "auc_A0",

                "ap_test":
                    "ap_A0",

                "logloss_test":
                    "logloss_A0",
            }
        )
    )

    x = (
        metricas
        .merge(
            base,
            on="seed",
            how="left",
        )
        .copy()
    )

    x["delta_auc_vs_A0"] = (
        x["auc_test"]
        - x["auc_A0"]
    )

    x["delta_ap_vs_A0"] = (
        x["ap_test"]
        - x["ap_A0"]
    )

    # En logloss MENOR es mejor.
    # Positivo = mejora respecto A0.
    x["mejora_logloss_vs_A0"] = (
        x["logloss_A0"]
        - x["logloss_test"]
    )

    resumen = (
        x.groupby(
            "experimento",
            as_index=False,
        )
        .agg(
            delta_auc_mean=(
                "delta_auc_vs_A0",
                "mean",
            ),

            delta_auc_std=(
                "delta_auc_vs_A0",
                "std",
            ),

            delta_auc_min=(
                "delta_auc_vs_A0",
                "min",
            ),

            delta_auc_max=(
                "delta_auc_vs_A0",
                "max",
            ),

            delta_ap_mean=(
                "delta_ap_vs_A0",
                "mean",
            ),

            delta_ap_std=(
                "delta_ap_vs_A0",
                "std",
            ),

            mejora_logloss_mean=(
                "mejora_logloss_vs_A0",
                "mean",
            ),

            semillas_mejor_auc=(
                "delta_auc_vs_A0",
                lambda s:
                    int(
                        (s > 0).sum()
                    ),
            ),

            semillas_mejor_ap=(
                "delta_ap_vs_A0",
                lambda s:
                    int(
                        (s > 0).sum()
                    ),
            ),
        )
    )

    orden = {
        "A0_t0": 0,
        "A1_t0_delta1": 1,
        "A2_t0_media2": 2,
        "A3_t0_delta1_media2": 3,
    }

    resumen["_orden"] = (
        resumen["experimento"]
        .map(orden)
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

    print(
        resumen.to_string(
            index=False
        )
    )

    return resumen


# =====================================================================
# IMPORTANCIAS
# =====================================================================

def resumir_importancias(
    importancias: pd.DataFrame,
) -> pd.DataFrame:

    titulo("IMPORTANCIA DE FEATURES")

    resumen = (
        importancias
        .groupby(
            [
                "experimento",
                "feature",
                "variable",
                "transformacion",
            ],
            as_index=False,
            dropna=False,
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

            split_mean=(
                "split",
                "mean",
            ),
        )
    )

    resumen["rank_gain"] = (
        resumen
        .groupby(
            "experimento"
        )["gain_mean"]
        .rank(
            method="dense",
            ascending=False,
        )
    )

    resumen = (
        resumen
        .sort_values(
            [
                "experimento",
                "rank_gain",
                "feature",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    for experimento in sorted(
        resumen["experimento"]
        .unique()
    ):

        print()
        print(
            f"TOP 15 — {experimento}"
        )

        x = (
            resumen[
                resumen["experimento"]
                == experimento
            ]
            .head(15)
        )

        print(
            x[
                [
                    "rank_gain",
                    "feature",
                    "gain_share_mean",
                ]
            ]
            .to_string(
                index=False
            )
        )

    return resumen


# =====================================================================
# IMPORTANCIA POR FAMILIA DE TRANSFORMACIÓN
# =====================================================================

def importancia_transformaciones(
    importancias: pd.DataFrame,
) -> pd.DataFrame:

    titulo("IMPORTANCIA POR TRANSFORMACIÓN")

    x = (
        importancias
        .groupby(
            [
                "experimento",
                "seed",
                "transformacion",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            gain_share=(
                "gain_share",
                "sum",
            )
        )
    )

    resumen = (
        x.groupby(
            [
                "experimento",
                "transformacion",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
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

    print(
        resumen.to_string(
            index=False
        )
    )

    return resumen


# =====================================================================
# PREDICCIÓN PROMEDIO POR EXPERIMENTO
# =====================================================================

def resumir_predicciones(
    predicciones: pd.DataFrame,
) -> pd.DataFrame:

    titulo("ENSEMBLE DE SEMILLAS — DIAGNÓSTICO")

    ensemble = (
        predicciones
        .groupby(
            [
                "numero_de_cliente",
                "foto_ancla",
                "grupo",
                "target",
                "experimento",
            ],
            as_index=False,
        )
        .agg(
            prob_mean=(
                "prob",
                "mean",
            ),

            prob_std=(
                "prob",
                "std",
            ),
        )
    )

    filas = []

    for experimento, g in (
        ensemble.groupby(
            "experimento"
        )
    ):

        auc = safe_auc(
            g["target"],
            g["prob_mean"],
        )

        ap = safe_ap(
            g["target"],
            g["prob_mean"],
        )

        ll = safe_logloss(
            g["target"],
            g["prob_mean"],
        )

        filas.append(
            {
                "experimento":
                    experimento,

                "auc_ensemble":
                    auc,

                "ap_ensemble":
                    ap,

                "logloss_ensemble":
                    ll,

                "prob_std_mean":
                    float(
                        g["prob_std"]
                        .mean()
                    ),
            }
        )

    res = pd.DataFrame(
        filas
    )

    print(
        res.to_string(
            index=False
        )
    )

    return ensemble


# =====================================================================
# COLUMNAS DE LOS MODELOS
# =====================================================================

def construir_tabla_columnas(
    infos: list[dict],
) -> pd.DataFrame:

    filas = []

    for info in infos:

        exp = info[
            "experimento"
        ]

        declaradas = set(
            info[
                "columnas_declaradas"
            ]
        )

        usadas = set(
            info[
                "columnas_usadas"
            ]
        )

        eliminadas = set(
            info[
                "columnas_eliminadas"
            ]
        )

        for feature in sorted(
            declaradas
        ):

            filas.append(
                {
                    "experimento":
                        exp,

                    "feature":
                        feature,

                    "variable":
                        variable_original(
                            feature
                        ),

                    "transformacion":
                        detectar_transformacion(
                            feature
                        ),

                    "declarada":
                        True,

                    "usada":
                        feature
                        in usadas,

                    "eliminada_train":
                        feature
                        in eliminadas,
                }
            )

    return pd.DataFrame(
        filas
    )


# =====================================================================
# RESUMEN TXT
# =====================================================================

def escribir_resumen(
    resumen_modelos: pd.DataFrame,
    comparacion: pd.DataFrame,
    imp_transformaciones: pd.DataFrame,
    columnas_modelos: pd.DataFrame,
) -> None:

    lines = []

    lines.append(
        "Z508 — VALIDACIÓN PREDICTIVA TEMPORAL"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        f"TRAIN: {TRAIN_FOTO}"
    )

    lines.append(
        f"TEST : {TEST_FOTO}"
    )

    lines.append("")

    lines.append(
        "Objetivo:"
    )

    lines.append(
        "evaluar si agregar historia reciente mejora "
        "el ranking predictivo respecto de t0."
    )

    lines.append("")

    lines.append(
        "MODELOS"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        resumen_modelos.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "COMPARACIÓN CONTRA A0"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        comparacion.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "IMPORTANCIA POR TRANSFORMACIÓN"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        imp_transformaciones.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "COLUMNAS UTILIZADAS"
    )

    lines.append(
        "-" * 72
    )

    colres = (
        columnas_modelos
        .groupby(
            "experimento"
        )
        .agg(
            declaradas=(
                "declarada",
                "sum",
            ),
            usadas=(
                "usada",
                "sum",
            ),
            eliminadas=(
                "eliminada_train",
                "sum",
            ),
        )
    )

    lines.append(
        colres.to_string()
    )

    lines.append("")

    lines.append(
        "INTERPRETACIÓN"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        "A0 es el baseline que utiliza únicamente "
        "la observación actual."
    )

    lines.append(
        "A1 agrega cambios respecto del mes anterior."
    )

    lines.append(
        "A2 agrega el nivel medio de las últimas "
        "dos observaciones."
    )

    lines.append(
        "A3 combina nivel actual, cambio y media reciente."
    )

    lines.append("")

    lines.append(
        "Una mejora consistente de A1/A2/A3 respecto "
        "de A0 en TEST constituye evidencia de que la "
        "historia reciente aporta información predictiva "
        "adicional."
    )

    lines.append("")

    lines.append(
        "Este experimento no optimiza threshold ni calcula "
        "todavía la ganancia económica de competencia."
    )

    OUT_DIR.joinpath(
        "resumen_z508.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    inicio_total = time.time()

    titulo(
        "Z508 — VALIDACIÓN PREDICTIVA TEMPORAL"
    )

    print(
        """
Pregunta experimental:

¿La historia inmediatamente anterior del cliente
agrega capacidad predictiva respecto de utilizar
solamente la observación actual?

TRAIN = 202104
TEST  = 202105

El modelo y los hiperparámetros permanecen fijos.
Sólo cambia el conjunto de features.
"""
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Carga
    # -------------------------------------------------------------

    df = cargar_features()

    # -------------------------------------------------------------
    # Validaciones
    # -------------------------------------------------------------

    validar_estructura(
        df
    )

    df = construir_target(
        df
    )

    train, test = (
        construir_train_test(
            df
        )
    )

    # -------------------------------------------------------------
    # Diseños
    # -------------------------------------------------------------

    diseños = (
        construir_diseno_features(
            df
        )
    )

    validar_disenos(
        diseños
    )

    # -------------------------------------------------------------
    # Entrenamiento
    # -------------------------------------------------------------

    titulo(
        "ENTRENAMIENTO LIGHTGBM"
    )

    print(
        "Seeds:",
        SEEDS,
    )

    print()
    print(
        "Parámetros:"
    )

    print(
        json.dumps(
            LGB_PARAMS,
            indent=2,
        )
    )

    todas_metricas = []
    todas_predicciones = []
    todas_importancias = []
    infos_columnas = []

    for nombre, columnas in (
        diseños.items()
    ):

        (
            metricas,
            predicciones,
            importancias,
            info_columnas,
        ) = entrenar_experimento(
            nombre,
            columnas,
            train,
            test,
        )

        todas_metricas.extend(
            metricas
        )

        todas_predicciones.extend(
            predicciones
        )

        todas_importancias.extend(
            importancias
        )

        infos_columnas.append(
            info_columnas
        )

    # -------------------------------------------------------------
    # Consolidación
    # -------------------------------------------------------------

    metricas_df = pd.DataFrame(
        todas_metricas
    )

    predicciones_df = pd.concat(
        todas_predicciones,
        ignore_index=True,
    )

    importancias_df = pd.concat(
        todas_importancias,
        ignore_index=True,
    )

    columnas_df = (
        construir_tabla_columnas(
            infos_columnas
        )
    )

    # -------------------------------------------------------------
    # Resúmenes
    # -------------------------------------------------------------

    resumen_modelos = (
        resumir_metricas(
            metricas_df
        )
    )

    comparacion = (
        comparar_baseline(
            metricas_df
        )
    )

    importancia_resumen = (
        resumir_importancias(
            importancias_df
        )
    )

    imp_transformaciones = (
        importancia_transformaciones(
            importancias_df
        )
    )

    ensemble = (
        resumir_predicciones(
            predicciones_df
        )
    )

    # -------------------------------------------------------------
    # Exportación
    # -------------------------------------------------------------

    titulo(
        "EXPORTACIÓN"
    )

    metricas_df.to_csv(
        OUT_DIR
        / "metricas_corridas.csv",
        index=False,
    )

    resumen_modelos.to_csv(
        OUT_DIR
        / "resumen_modelos.csv",
        index=False,
    )

    comparacion.to_csv(
        OUT_DIR
        / "comparacion_baseline.csv",
        index=False,
    )

    predicciones_df.to_csv(
        OUT_DIR
        / "predicciones_test.csv",
        index=False,
    )

    ensemble.to_csv(
        OUT_DIR
        / "predicciones_ensemble.csv",
        index=False,
    )

    importancias_df.to_csv(
        OUT_DIR
        / "importancia_gain.csv",
        index=False,
    )

    importancia_resumen.to_csv(
        OUT_DIR
        / "importancia_resumen.csv",
        index=False,
    )

    imp_transformaciones.to_csv(
        OUT_DIR
        / "importancia_transformaciones.csv",
        index=False,
    )

    columnas_df.to_csv(
        OUT_DIR
        / "columnas_modelos.csv",
        index=False,
    )

    escribir_resumen(
        resumen_modelos,
        comparacion,
        imp_transformaciones,
        columnas_df,
    )

    archivos = [
        "metricas_corridas.csv",
        "resumen_modelos.csv",
        "comparacion_baseline.csv",
        "predicciones_test.csv",
        "predicciones_ensemble.csv",
        "importancia_gain.csv",
        "importancia_resumen.csv",
        "importancia_transformaciones.csv",
        "columnas_modelos.csv",
        "resumen_z508.txt",
    ]

    for nombre in archivos:

        p = OUT_DIR / nombre

        print(
            f"OK  {nombre:38s} "
            f"{p.stat().st_size:>12,} bytes"
        )

    # -------------------------------------------------------------
    # Resumen final
    # -------------------------------------------------------------

    titulo(
        "RESULTADO FINAL Z508"
    )

    print(
        resumen_modelos[
            [
                "experimento",
                "n_features",
                "auc_test_mean",
                "auc_test_std",
                "ap_test_mean",
                "logloss_test_mean",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Cambio respecto de A0:"
    )

    print(
        comparacion[
            [
                "experimento",
                "delta_auc_mean",
                "delta_auc_std",
                "delta_ap_mean",
                "mejora_logloss_mean",
                "semillas_mejor_auc",
            ]
        ].to_string(
            index=False
        )
    )

    elapsed = (
        time.time()
        - inicio_total
    )

    print()
    print(
        f"Tiempo total: "
        f"{elapsed:.2f} s"
    )

    print()
    print(
        "Z508 FINALIZADO CORRECTAMENTE."
    )

    print()
    print(
        f"Resultados:\n{OUT_DIR}"
    )


# =====================================================================
# ENTRYPOINT
# =====================================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        titulo(
            "ERROR EN Z508"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        traceback.print_exc()

        sys.exit(1)