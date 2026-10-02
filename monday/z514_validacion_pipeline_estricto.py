#!/usr/bin/env python3

"""
Z514 - Validacion estricta del pipeline predictivo pre-BAJA+2.

Objetivo
--------
Evaluar un pipeline out-of-time + out-of-client en el cual TODA decision
dependiente de los datos se aprende exclusivamente dentro del TRAIN de
cada split.

Para cada split_seed:

1. TRAIN temporal = 202104
2. TEST temporal  = 202105
3. Los clientes FIEL se particionan antes de cualquier seleccion.
4. No existe interseccion de clientes TRAIN/TEST.
5. Sobre TRAIN solamente:
   a. se determinan columnas t0 y media2 validas;
   b. se rankean las media2 condicionadas por t0;
   c. se seleccionan Top40 media2;
   d. se construye C0 = t0 + Top40 media2;
   e. se rankea conjuntamente C0.
6. Se evaluan:
   - Top70 de C0
   - Top100 de C0
   - C0 completo
7. TEST no participa en ninguna seleccion ni ranking.

Nota metodologica
-----------------
Los tamanos 40, 70 y 100 provienen de experimentos anteriores y se
consideran configuraciones predefinidas para este ejercicio. Por lo tanto,
Z514 elimina la contaminacion de seleccion por clientes FIEL retenidos,
pero mayo 202105 no debe describirse como un holdout historicamente virgen.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


# ======================================================================
# CONFIGURACION
# ======================================================================

INPUT = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "prebaja_features/features_prebaja.csv"
)

OUT_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "validacion_pipeline_estricto"
)

TRAIN_FOTO = 202104
TEST_FOTO = 202105

SPLIT_SEEDS = [
    100,
    200,
    300,
    400,
    500,
]

# Ranking interno: se conserva la filosofia de Z509/Z511.
RANKING_SEEDS = [
    100,
    200,
    300,
    400,
    500,
]

N_MEDIA_BASE = 40

TOP_COMPACTOS = [
    70,
    100,
]

MODEL_PARAMS = {
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

def titulo(texto: str) -> None:
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def subtitulo(texto: str) -> None:
    print()
    print("-" * 78)
    print(texto)
    print("-" * 78)


def nuevo_modelo(seed: int):
    return lgb.LGBMClassifier(
        **MODEL_PARAMS,
        random_state=seed,
    )


def preparar_X(
    df: pd.DataFrame,
    columnas: list[str],
) -> pd.DataFrame:
    return (
        df[columnas]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .astype("float32")
    )


def detectar_columnas(
    columnas: list[str],
    sufijo: str,
) -> list[str]:
    return sorted(
        c
        for c in columnas
        if c.endswith(sufijo)
    )


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

        n = int(x.notna().sum())

        if n == 0:
            eliminadas.append(
                {
                    "feature": col,
                    "motivo": "sin_datos_train",
                }
            )
            continue

        nunique = int(
            x.nunique(dropna=True)
        )

        if nunique <= 1:
            eliminadas.append(
                {
                    "feature": col,
                    "motivo": "constante_train",
                }
            )
            continue

        validas.append(col)

    return validas, eliminadas


# ======================================================================
# CARGA
# ======================================================================

def cargar_datos() -> pd.DataFrame:
    titulo("CARGA DE DATOS")

    if not INPUT.exists():
        raise FileNotFoundError(
            f"No existe:\n{INPUT}"
        )

    print(f"Input: {INPUT}")

    df = pd.read_csv(INPUT)

    print(
        f"Dimensiones: "
        f"{df.shape[0]:,} x {df.shape[1]:,}"
    )

    requeridas = {
        "numero_de_cliente",
        "foto_ancla",
        "grupo",
    }

    faltantes = requeridas - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {faltantes}"
        )

    df["target"] = (
        df["grupo"]
        .eq("BAJA+2")
        .astype("int8")
    )

    print()
    print("Distribucion por foto y grupo:")
    print(
        df.groupby(
            ["foto_ancla", "grupo"]
        )
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    return df


# ======================================================================
# SPLIT OUT-OF-TIME + OUT-OF-CLIENT
# ======================================================================

def construir_split(
    df: pd.DataFrame,
    seed: int,
):
    base_train = (
        df[
            df["foto_ancla"]
            == TRAIN_FOTO
        ]
        .copy()
    )

    base_test = (
        df[
            df["foto_ancla"]
            == TEST_FOTO
        ]
        .copy()
    )

    baja_train = (
        base_train[
            base_train["target"] == 1
        ]
        .copy()
    )

    baja_test = (
        base_test[
            base_test["target"] == 1
        ]
        .copy()
    )

    fiel_train_base = (
        base_train[
            base_train["target"] == 0
        ]
        .copy()
    )

    fiel_test_base = (
        base_test[
            base_test["target"] == 0
        ]
        .copy()
    )

    ids_fiel_train = set(
        fiel_train_base[
            "numero_de_cliente"
        ].tolist()
    )

    ids_fiel_test = set(
        fiel_test_base[
            "numero_de_cliente"
        ].tolist()
    )

    if ids_fiel_train != ids_fiel_test:
        raise ValueError(
            "Los universos FIEL de TRAIN y TEST "
            "no son identicos."
        )

    ids_fiel = np.array(
        sorted(ids_fiel_train)
    )

    if len(ids_fiel) != 4067:
        raise ValueError(
            "Se esperaban 4067 clientes FIEL; "
            f"se encontraron {len(ids_fiel)}."
        )

    n_baja_train = len(baja_train)
    n_baja_test = len(baja_test)

    n_fiel_train = round(
        len(ids_fiel)
        * n_baja_train
        / (n_baja_train + n_baja_test)
    )

    rng = np.random.default_rng(seed)

    ids_permutados = rng.permutation(
        ids_fiel
    )

    ids_train = set(
        ids_permutados[
            :n_fiel_train
        ].tolist()
    )

    ids_test = set(
        ids_permutados[
            n_fiel_train:
        ].tolist()
    )

    fiel_train = (
        fiel_train_base[
            fiel_train_base[
                "numero_de_cliente"
            ].isin(ids_train)
        ]
        .copy()
    )

    fiel_test = (
        fiel_test_base[
            fiel_test_base[
                "numero_de_cliente"
            ].isin(ids_test)
        ]
        .copy()
    )

    train = pd.concat(
        [
            baja_train,
            fiel_train,
        ],
        ignore_index=True,
    )

    test = pd.concat(
        [
            baja_test,
            fiel_test,
        ],
        ignore_index=True,
    )

    clientes_train = set(
        train["numero_de_cliente"]
        .tolist()
    )

    clientes_test = set(
        test["numero_de_cliente"]
        .tolist()
    )

    interseccion = (
        clientes_train
        & clientes_test
    )

    inter_baja = (
        set(
            baja_train[
                "numero_de_cliente"
            ].tolist()
        )
        & set(
            baja_test[
                "numero_de_cliente"
            ].tolist()
        )
    )

    inter_fiel = (
        ids_train
        & ids_test
    )

    if interseccion:
        raise ValueError(
            "Hay clientes compartidos "
            "entre TRAIN y TEST."
        )

    if inter_baja:
        raise ValueError(
            "Hay BAJA+2 compartidos "
            "entre TRAIN y TEST."
        )

    if inter_fiel:
        raise ValueError(
            "Hay FIEL compartidos "
            "entre TRAIN y TEST."
        )

    if len(baja_train) != 865:
        raise ValueError(
            f"BAJA+2 TRAIN inesperados: "
            f"{len(baja_train)}"
        )

    if len(baja_test) != 1097:
        raise ValueError(
            f"BAJA+2 TEST inesperados: "
            f"{len(baja_test)}"
        )

    if len(fiel_train) != 1793:
        raise ValueError(
            f"FIEL TRAIN inesperados: "
            f"{len(fiel_train)}"
        )

    if len(fiel_test) != 2274:
        raise ValueError(
            f"FIEL TEST inesperados: "
            f"{len(fiel_test)}"
        )

    if train["target"].nunique() != 2:
        raise ValueError(
            "TRAIN no contiene ambas clases."
        )

    if test["target"].nunique() != 2:
        raise ValueError(
            "TEST no contiene ambas clases."
        )

    diagnostico = {
        "split_seed": seed,
        "n_train": len(train),
        "n_test": len(test),
        "baja_train": int(
            train["target"].sum()
        ),
        "fiel_train": int(
            (train["target"] == 0).sum()
        ),
        "baja_test": int(
            test["target"].sum()
        ),
        "fiel_test": int(
            (test["target"] == 0).sum()
        ),
        "clientes_train":
            train[
                "numero_de_cliente"
            ].nunique(),
        "clientes_test":
            test[
                "numero_de_cliente"
            ].nunique(),
        "interseccion_clientes":
            len(interseccion),
        "prevalencia_train":
            float(train["target"].mean()),
        "prevalencia_test":
            float(test["target"].mean()),
    }

    return (
        train,
        test,
        diagnostico,
    )


# ======================================================================
# RANKING POR GAIN — SOLO TRAIN
# ======================================================================

def ranking_gain_train(
    train: pd.DataFrame,
    columnas_modelo: list[str],
    columnas_rankear: list[str],
    split_seed: int,
    etapa: str,
):
    columnas_modelo = list(
        dict.fromkeys(
            columnas_modelo
        )
    )

    columnas_rankear = list(
        dict.fromkeys(
            columnas_rankear
        )
    )

    permitidas = set(
        columnas_rankear
    )

    X_train = preparar_X(
        train,
        columnas_modelo,
    )

    y_train = (
        train["target"]
        .to_numpy(dtype="int8")
    )

    registros = []

    for ranking_seed in RANKING_SEEDS:
        model = nuevo_modelo(
            ranking_seed
        )

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
            share = gain / total
        else:
            share = np.zeros_like(
                gain,
                dtype=float,
            )

        tmp = pd.DataFrame(
            {
                "split_seed":
                    split_seed,
                "etapa":
                    etapa,
                "ranking_seed":
                    ranking_seed,
                "feature":
                    columnas_modelo,
                "gain":
                    gain,
                "gain_share":
                    share,
            }
        )

        tmp = (
            tmp[
                tmp["feature"]
                .isin(permitidas)
            ]
            .copy()
        )

        tmp = (
            tmp.sort_values(
                [
                    "gain_share",
                    "feature",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        tmp["rank_seed"] = (
            np.arange(
                1,
                len(tmp) + 1,
            )
        )

        registros.append(tmp)

    ranking_seed_df = pd.concat(
        registros,
        ignore_index=True,
    )

    resumen = (
        ranking_seed_df
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
            rank_seed_mean=(
                "rank_seed",
                "mean",
            ),
            rank_seed_std=(
                "rank_seed",
                "std",
            ),
        )
    )

    resumen = (
        resumen.sort_values(
            [
                "gain_share_mean",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    resumen["rank"] = (
        np.arange(
            1,
            len(resumen) + 1,
        )
    )

    resumen.insert(
        0,
        "etapa",
        etapa,
    )

    resumen.insert(
        0,
        "split_seed",
        split_seed,
    )

    return (
        resumen,
        ranking_seed_df,
    )


# ======================================================================
# PIPELINE DE SELECCION — SOLO TRAIN
# ======================================================================

def seleccionar_features_train(
    train: pd.DataFrame,
    split_seed: int,
):
    columnas = train.columns.tolist()

    t0_declaradas = detectar_columnas(
        columnas,
        "__t0",
    )

    media_declaradas = detectar_columnas(
        columnas,
        "__media2",
    )

    (
        t0_validas,
        t0_eliminadas,
    ) = columnas_validas_train(
        train,
        t0_declaradas,
    )

    (
        media_validas,
        media_eliminadas,
    ) = columnas_validas_train(
        train,
        media_declaradas,
    )

    if len(t0_validas) == 0:
        raise ValueError(
            "No quedaron features t0 validas."
        )

    if len(media_validas) < N_MEDIA_BASE:
        raise ValueError(
            "No hay suficientes media2 "
            f"validas para Top{N_MEDIA_BASE}: "
            f"{len(media_validas)}"
        )

    eliminadas = []

    for fila in t0_eliminadas:
        eliminadas.append(
            {
                "split_seed":
                    split_seed,
                "familia":
                    "t0",
                **fila,
            }
        )

    for fila in media_eliminadas:
        eliminadas.append(
            {
                "split_seed":
                    split_seed,
                "familia":
                    "media2",
                **fila,
            }
        )

    # --------------------------------------------------------------
    # Etapa 1: ranking media2 condicionado por t0
    # --------------------------------------------------------------

    ranking_media2, ranking_media2_seed = (
        ranking_gain_train(
            train=train,
            columnas_modelo=(
                t0_validas
                + media_validas
            ),
            columnas_rankear=
                media_validas,
            split_seed=split_seed,
            etapa="media2",
        )
    )

    media2_top40 = (
        ranking_media2[
            "feature"
        ]
        .head(N_MEDIA_BASE)
        .tolist()
    )

    if len(media2_top40) != N_MEDIA_BASE:
        raise ValueError(
            "No fue posible seleccionar "
            "40 features media2."
        )

    # --------------------------------------------------------------
    # C0 aprendido dentro de este TRAIN
    # --------------------------------------------------------------

    c0 = list(
        dict.fromkeys(
            t0_validas
            + media2_top40
        )
    )

    # --------------------------------------------------------------
    # Etapa 2: ranking conjunto C0
    # --------------------------------------------------------------

    ranking_c0, ranking_c0_seed = (
        ranking_gain_train(
            train=train,
            columnas_modelo=c0,
            columnas_rankear=c0,
            split_seed=split_seed,
            etapa="c0",
        )
    )

    orden_c0 = (
        ranking_c0[
            "feature"
        ]
        .tolist()
    )

    if set(orden_c0) != set(c0):
        raise ValueError(
            "Ranking C0 no contiene "
            "exactamente las features de C0."
        )

    if len(orden_c0) < max(
        TOP_COMPACTOS
    ):
        raise ValueError(
            "C0 tiene menos features que "
            "el mayor Top solicitado: "
            f"{len(orden_c0)}"
        )

    disenos = {
        "G0_c0_top70":
            orden_c0[:70],
        "G1_c0_top100":
            orden_c0[:100],
        "G2_c0_all":
            list(c0),
    }

    seleccion = []

    for posicion, feature in enumerate(
        media2_top40,
        start=1,
    ):
        seleccion.append(
            {
                "split_seed":
                    split_seed,
                "seleccion":
                    "media2_top40",
                "posicion":
                    posicion,
                "feature":
                    feature,
            }
        )

    for nombre, cols in disenos.items():
        for posicion, feature in enumerate(
            cols,
            start=1,
        ):
            seleccion.append(
                {
                    "split_seed":
                        split_seed,
                    "seleccion":
                        nombre,
                    "posicion":
                        posicion,
                    "feature":
                        feature,
                }
            )

    diagnostico = {
        "split_seed":
            split_seed,
        "t0_declaradas":
            len(t0_declaradas),
        "t0_validas":
            len(t0_validas),
        "t0_eliminadas":
            len(t0_eliminadas),
        "media2_declaradas":
            len(media_declaradas),
        "media2_validas":
            len(media_validas),
        "media2_eliminadas":
            len(media_eliminadas),
        "n_media2_top40":
            len(media2_top40),
        "n_c0":
            len(c0),
        "n_c0_t0":
            sum(
                c.endswith("__t0")
                for c in c0
            ),
        "n_c0_media2":
            sum(
                c.endswith("__media2")
                for c in c0
            ),
    }

    return {
        "disenos":
            disenos,
        "seleccion":
            seleccion,
        "eliminadas":
            eliminadas,
        "ranking_media2":
            ranking_media2,
        "ranking_media2_seed":
            ranking_media2_seed,
        "ranking_c0":
            ranking_c0,
        "ranking_c0_seed":
            ranking_c0_seed,
        "diagnostico":
            diagnostico,
    }


# ======================================================================
# EVALUACION ESTRICTA
# ======================================================================

def evaluar_pipeline(
    df: pd.DataFrame,
):
    titulo(
        "EVALUACION ESTRICTA "
        "OUT-OF-TIME + OUT-OF-CLIENT"
    )

    metricas = []
    predicciones = []
    importancias = []
    diagnosticos_split = []
    diagnosticos_features = []
    selecciones = []
    eliminadas = []

    rankings_resumen = []
    rankings_seed = []

    for split_seed in SPLIT_SEEDS:
        subtitulo(
            f"SPLIT {split_seed}"
        )

        (
            train,
            test,
            diagnostico_split,
        ) = construir_split(
            df=df,
            seed=split_seed,
        )

        diagnosticos_split.append(
            diagnostico_split
        )

        print(
            f"TRAIN: {len(train):,} "
            f"(BAJA+2={int(train['target'].sum()):,}, "
            f"FIEL={int((train['target'] == 0).sum()):,})"
        )

        print(
            f"TEST : {len(test):,} "
            f"(BAJA+2={int(test['target'].sum()):,}, "
            f"FIEL={int((test['target'] == 0).sum()):,})"
        )

        print(
            "Interseccion clientes: "
            f"{diagnostico_split['interseccion_clientes']}"
        )

        inicio_sel = time.time()

        seleccion_train = (
            seleccionar_features_train(
                train=train,
                split_seed=split_seed,
            )
        )

        segundos_sel = (
            time.time()
            - inicio_sel
        )

        diagnostico_features = (
            seleccion_train[
                "diagnostico"
            ].copy()
        )

        diagnostico_features[
            "segundos_seleccion"
        ] = segundos_sel

        diagnosticos_features.append(
            diagnostico_features
        )

        selecciones.extend(
            seleccion_train[
                "seleccion"
            ]
        )

        eliminadas.extend(
            seleccion_train[
                "eliminadas"
            ]
        )

        rankings_resumen.extend(
            [
                seleccion_train[
                    "ranking_media2"
                ],
                seleccion_train[
                    "ranking_c0"
                ],
            ]
        )

        rankings_seed.extend(
            [
                seleccion_train[
                    "ranking_media2_seed"
                ],
                seleccion_train[
                    "ranking_c0_seed"
                ],
            ]
        )

        print(
            "Features validas: "
            f"t0={diagnostico_features['t0_validas']}, "
            f"media2={diagnostico_features['media2_validas']}"
        )

        print(
            f"C0 aprendido: "
            f"{diagnostico_features['n_c0']} "
            "features"
        )

        disenos = (
            seleccion_train[
                "disenos"
            ]
        )

        y_train = (
            train["target"]
            .to_numpy(dtype="int8")
        )

        y_test = (
            test["target"]
            .to_numpy(dtype="int8")
        )

        for nombre, cols in disenos.items():
            print(
                f"{nombre:16s} "
                f"features={len(cols):3d} ...",
                end=" ",
                flush=True,
            )

            inicio_fit = time.time()

            X_train = preparar_X(
                train,
                cols,
            )

            X_test = preparar_X(
                test,
                cols,
            )

            # La seed externa controla tanto
            # el split como el modelo final.
            model = nuevo_modelo(
                split_seed
            )

            model.fit(
                X_train,
                y_train,
            )

            prob = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )

            auc = float(
                roc_auc_score(
                    y_test,
                    prob,
                )
            )

            ap = float(
                average_precision_score(
                    y_test,
                    prob,
                )
            )

            ll = float(
                log_loss(
                    y_test,
                    prob,
                )
            )

            segundos = (
                time.time()
                - inicio_fit
            )

            print(
                f"AUC={auc:.6f} "
                f"AP={ap:.6f} "
                f"LL={ll:.6f} "
                f"{segundos:.2f}s"
            )

            metricas.append(
                {
                    "split_seed":
                        split_seed,
                    "experimento":
                        nombre,
                    "n_features":
                        len(cols),
                    "auc":
                        auc,
                    "average_precision":
                        ap,
                    "logloss":
                        ll,
                    "segundos_fit":
                        segundos,
                }
            )

            predicciones.append(
                pd.DataFrame(
                    {
                        "split_seed":
                            split_seed,
                        "experimento":
                            nombre,
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
                            y_test,
                        "prob":
                            prob,
                    }
                )
            )

            gain = (
                model.booster_
                .feature_importance(
                    importance_type="gain"
                )
            )

            total_gain = float(
                gain.sum()
            )

            if total_gain > 0:
                gain_share = (
                    gain / total_gain
                )
            else:
                gain_share = (
                    np.zeros_like(
                        gain,
                        dtype=float,
                    )
                )

            importancias.append(
                pd.DataFrame(
                    {
                        "split_seed":
                            split_seed,
                        "experimento":
                            nombre,
                        "feature":
                            cols,
                        "gain":
                            gain,
                        "gain_share":
                            gain_share,
                    }
                )
            )

    return {
        "metricas":
            pd.DataFrame(metricas),
        "predicciones":
            pd.concat(
                predicciones,
                ignore_index=True,
            ),
        "importancias":
            pd.concat(
                importancias,
                ignore_index=True,
            ),
        "diagnosticos_split":
            pd.DataFrame(
                diagnosticos_split
            ),
        "diagnosticos_features":
            pd.DataFrame(
                diagnosticos_features
            ),
        "selecciones":
            pd.DataFrame(
                selecciones
            ),
        "eliminadas":
            pd.DataFrame(
                eliminadas
            ),
        "rankings_resumen":
            pd.concat(
                rankings_resumen,
                ignore_index=True,
            ),
        "rankings_seed":
            pd.concat(
                rankings_seed,
                ignore_index=True,
            ),
    }


# ======================================================================
# RESUMEN DE MODELOS
# ======================================================================

def resumir_modelos(
    metricas: pd.DataFrame,
) -> pd.DataFrame:
    resumen = (
        metricas
        .groupby(
            "experimento",
            as_index=False,
        )
        .agg(
            n_features_mean=(
                "n_features",
                "mean",
            ),
            n_features_min=(
                "n_features",
                "min",
            ),
            n_features_max=(
                "n_features",
                "max",
            ),
            auc_mean=(
                "auc",
                "mean",
            ),
            auc_std=(
                "auc",
                "std",
            ),
            auc_min=(
                "auc",
                "min",
            ),
            auc_max=(
                "auc",
                "max",
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
            logloss_std=(
                "logloss",
                "std",
            ),
        )
    )

    referencia = (
        resumen[
            resumen["experimento"]
            == "G2_c0_all"
        ]
    )

    if len(referencia) != 1:
        raise ValueError(
            "No se encontro exactamente "
            "una referencia G2_c0_all."
        )

    auc_ref = float(
        referencia[
            "auc_mean"
        ].iloc[0]
    )

    ap_ref = float(
        referencia[
            "ap_mean"
        ].iloc[0]
    )

    ll_ref = float(
        referencia[
            "logloss_mean"
        ].iloc[0]
    )

    resumen["delta_auc_vs_all"] = (
        resumen["auc_mean"]
        - auc_ref
    )

    resumen["delta_ap_vs_all"] = (
        resumen["ap_mean"]
        - ap_ref
    )

    resumen["delta_logloss_vs_all"] = (
        resumen["logloss_mean"]
        - ll_ref
    )

    return resumen


# ======================================================================
# ESTABILIDAD DE SELECCION
# ======================================================================

def frecuencia_seleccion(
    selecciones: pd.DataFrame,
) -> pd.DataFrame:
    total_splits = len(
        SPLIT_SEEDS
    )

    frecuencia = (
        selecciones[
            [
                "split_seed",
                "seleccion",
                "feature",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "seleccion",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            n_splits=(
                "split_seed",
                "nunique",
            )
        )
    )

    frecuencia[
        "frecuencia"
    ] = (
        frecuencia["n_splits"]
        / total_splits
    )

    frecuencia = (
        frecuencia.sort_values(
            [
                "seleccion",
                "n_splits",
                "feature",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    return frecuencia


def calcular_jaccard(
    selecciones: pd.DataFrame,
) -> pd.DataFrame:
    filas = []

    tipos = sorted(
        selecciones[
            "seleccion"
        ].unique()
    )

    for tipo in tipos:
        sub = (
            selecciones[
                selecciones[
                    "seleccion"
                ] == tipo
            ]
        )

        conjuntos = {}

        for seed in SPLIT_SEEDS:
            conjuntos[seed] = set(
                sub[
                    sub[
                        "split_seed"
                    ] == seed
                ][
                    "feature"
                ].tolist()
            )

        for seed_a, seed_b in (
            itertools.combinations(
                SPLIT_SEEDS,
                2,
            )
        ):
            a = conjuntos[seed_a]
            b = conjuntos[seed_b]

            union = a | b
            inter = a & b

            if len(union) == 0:
                jaccard = np.nan
            else:
                jaccard = (
                    len(inter)
                    / len(union)
                )

            filas.append(
                {
                    "seleccion":
                        tipo,
                    "seed_a":
                        seed_a,
                    "seed_b":
                        seed_b,
                    "n_a":
                        len(a),
                    "n_b":
                        len(b),
                    "interseccion":
                        len(inter),
                    "union":
                        len(union),
                    "jaccard":
                        jaccard,
                }
            )

    return pd.DataFrame(
        filas
    )


def resumir_jaccard(
    jaccard: pd.DataFrame,
) -> pd.DataFrame:
    return (
        jaccard
        .groupby(
            "seleccion",
            as_index=False,
        )
        .agg(
            jaccard_mean=(
                "jaccard",
                "mean",
            ),
            jaccard_std=(
                "jaccard",
                "std",
            ),
            jaccard_min=(
                "jaccard",
                "min",
            ),
            jaccard_max=(
                "jaccard",
                "max",
            ),
        )
    )


# ======================================================================
# IMPORTANCIA FINAL
# ======================================================================

def resumir_importancias(
    importancias: pd.DataFrame,
) -> pd.DataFrame:
    resumen = (
        importancias
        .groupby(
            [
                "experimento",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            n_splits=(
                "split_seed",
                "nunique",
            ),
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

    return (
        resumen.sort_values(
            [
                "experimento",
                "gain_share_mean",
                "feature",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


# ======================================================================
# RESUMEN TXT
# ======================================================================

def guardar_resumen_txt(
    resumen_modelos: pd.DataFrame,
    diagnosticos_split: pd.DataFrame,
    diagnosticos_features: pd.DataFrame,
    resumen_jaccard: pd.DataFrame,
    segundos_total: float,
):
    archivo = (
        OUT_DIR
        / "resumen_z514.txt"
    )

    with archivo.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "Z514 - VALIDACION ESTRICTA DEL PIPELINE\n"
        )
        f.write(
            "=" * 78 + "\n\n"
        )

        f.write(
            f"TRAIN temporal: {TRAIN_FOTO}\n"
        )
        f.write(
            f"TEST temporal : {TEST_FOTO}\n"
        )
        f.write(
            "Split seeds   : "
            + ", ".join(
                map(str, SPLIT_SEEDS)
            )
            + "\n"
        )
        f.write(
            "Ranking seeds : "
            + ", ".join(
                map(str, RANKING_SEEDS)
            )
            + "\n\n"
        )

        f.write(
            "REGLA METODOLOGICA\n"
        )
        f.write(
            "-" * 78 + "\n"
        )
        f.write(
            "El split de clientes ocurre antes de determinar columnas validas,\n"
        )
        f.write(
            "rankear media2, seleccionar Top40, construir C0 y rankear C0.\n"
        )
        f.write(
            "TEST no interviene en ninguna de esas decisiones.\n\n"
        )

        f.write(
            "Los tamanos Top40/Top70/Top100 fueron definidos en experimentos\n"
        )
        f.write(
            "anteriores; por ello mayo 202105 no constituye un holdout\n"
        )
        f.write(
            "historicamente virgen para seleccion de hiperparametros.\n\n"
        )

        f.write(
            "DIAGNOSTICO SPLITS\n"
        )
        f.write(
            "-" * 78 + "\n"
        )
        f.write(
            diagnosticos_split
            .to_string(index=False)
        )
        f.write("\n\n")

        f.write(
            "DIAGNOSTICO FEATURES\n"
        )
        f.write(
            "-" * 78 + "\n"
        )
        f.write(
            diagnosticos_features
            .to_string(index=False)
        )
        f.write("\n\n")

        f.write(
            "RESULTADOS\n"
        )
        f.write(
            "-" * 78 + "\n"
        )
        f.write(
            resumen_modelos
            .to_string(index=False)
        )
        f.write("\n\n")

        f.write(
            "ESTABILIDAD JACCARD\n"
        )
        f.write(
            "-" * 78 + "\n"
        )
        f.write(
            resumen_jaccard
            .to_string(index=False)
        )
        f.write("\n\n")

        f.write(
            f"Tiempo total: "
            f"{segundos_total:.2f} s\n"
        )


# ======================================================================
# MAIN
# ======================================================================

def main():
    inicio_total = time.time()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    titulo(
        "Z514 — VALIDACION ESTRICTA "
        "DEL PIPELINE"
    )

    print(
        f"TRAIN = {TRAIN_FOTO}"
    )
    print(
        f"TEST  = {TEST_FOTO}"
    )
    print(
        "Split seeds = "
        + ", ".join(
            map(str, SPLIT_SEEDS)
        )
    )
    print(
        "Ranking seeds = "
        + ", ".join(
            map(str, RANKING_SEEDS)
        )
    )

    df = cargar_datos()

    resultados = evaluar_pipeline(
        df
    )

    metricas = resultados[
        "metricas"
    ]

    resumen_modelos = (
        resumir_modelos(
            metricas
        )
    )

    frecuencia = (
        frecuencia_seleccion(
            resultados[
                "selecciones"
            ]
        )
    )

    jaccard = calcular_jaccard(
        resultados[
            "selecciones"
        ]
    )

    resumen_jaccard = (
        resumir_jaccard(
            jaccard
        )
    )

    importancia_resumen = (
        resumir_importancias(
            resultados[
                "importancias"
            ]
        )
    )

    # --------------------------------------------------------------
    # Guardado
    # --------------------------------------------------------------

    metricas.to_csv(
        OUT_DIR
        / "metricas_corridas.csv",
        index=False,
    )

    resumen_modelos.to_csv(
        OUT_DIR
        / "resumen_modelos.csv",
        index=False,
    )

    resultados[
        "predicciones"
    ].to_csv(
        OUT_DIR
        / "predicciones_test.csv",
        index=False,
    )

    resultados[
        "importancias"
    ].to_csv(
        OUT_DIR
        / "importancia_gain.csv",
        index=False,
    )

    importancia_resumen.to_csv(
        OUT_DIR
        / "importancia_resumen.csv",
        index=False,
    )

    resultados[
        "diagnosticos_split"
    ].to_csv(
        OUT_DIR
        / "diagnostico_splits.csv",
        index=False,
    )

    resultados[
        "diagnosticos_features"
    ].to_csv(
        OUT_DIR
        / "diagnostico_features.csv",
        index=False,
    )

    resultados[
        "selecciones"
    ].to_csv(
        OUT_DIR
        / "features_seleccionadas_por_split.csv",
        index=False,
    )

    frecuencia.to_csv(
        OUT_DIR
        / "frecuencia_seleccion_features.csv",
        index=False,
    )

    jaccard.to_csv(
        OUT_DIR
        / "jaccard_selecciones.csv",
        index=False,
    )

    resumen_jaccard.to_csv(
        OUT_DIR
        / "resumen_jaccard.csv",
        index=False,
    )

    resultados[
        "rankings_resumen"
    ].to_csv(
        OUT_DIR
        / "rankings_train_por_split.csv",
        index=False,
    )

    resultados[
        "rankings_seed"
    ].to_csv(
        OUT_DIR
        / "rankings_train_por_split_y_seed.csv",
        index=False,
    )

    if len(
        resultados[
            "eliminadas"
        ]
    ) > 0:
        resultados[
            "eliminadas"
        ].to_csv(
            OUT_DIR
            / "columnas_eliminadas_por_split.csv",
            index=False,
        )

    segundos_total = (
        time.time()
        - inicio_total
    )

    guardar_resumen_txt(
        resumen_modelos=
            resumen_modelos,
        diagnosticos_split=
            resultados[
                "diagnosticos_split"
            ],
        diagnosticos_features=
            resultados[
                "diagnosticos_features"
            ],
        resumen_jaccard=
            resumen_jaccard,
        segundos_total=
            segundos_total,
    )

    titulo(
        "RESUMEN Z514"
    )

    print(
        resumen_modelos
        .to_string(index=False)
    )

    print()
    print(
        "Estabilidad de seleccion:"
    )
    print(
        resumen_jaccard
        .to_string(index=False)
    )

    print()
    print(
        f"Tiempo total: "
        f"{segundos_total:.2f} s"
    )

    print()
    print(
        "Z514 FINALIZADO CORRECTAMENTE."
    )


if __name__ == "__main__":
    main()