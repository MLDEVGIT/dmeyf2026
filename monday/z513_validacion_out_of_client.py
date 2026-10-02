#!/usr/bin/env python3

"""
Z511 - Compactacion del diseño C0.

Objetivo
--------
Determinar cuantas features del diseño C0 necesitamos para conservar
su capacidad predictiva out-of-time.

C0 proviene de Z509/Z510:

    todas las t0 validas
    +
    Top40 media2 seleccionadas condicionalmente a t0

Diseño temporal
---------------
TRAIN = 202104
TEST  = 202105

Toda seleccion de variables se realiza EXCLUSIVAMENTE con TRAIN.

Experimentos
------------
D0 = Top10 features de C0
D1 = Top20
D2 = Top40
D3 = Top80
D4 = Top120
D5 = Top160
D6 = All C0

El ranking conjunto de C0 se obtiene promediando gain_share
de LightGBM sobre las cinco semillas.

Control de regresion
--------------------
D6 debe reproducir C0 de Z510 / B3 de Z509.
"""

from __future__ import annotations

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
    "validacion_out_of_client"
)

Z511_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "compactacion_c0"
)

Z510_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "seleccion_delta1"
)

Z509_DIR = Path(
    "/data/dmeyf/datasets/evaluacion_clusters/"
    "seleccion_media2"
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

N_MEDIA_BASE = 40

TOP_N = [
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
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
) -> list[str]:

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

    return validas


# ======================================================================
# CARGA
# ======================================================================

def cargar_datos():

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
        "foto_ancla",
        "grupo",
    }

    faltantes = (
        requeridas
        - set(df.columns)
    )

    if faltantes:
        raise ValueError(
            f"Faltan columnas: {faltantes}"
        )

    df["target"] = (
        df["grupo"]
        .eq("BAJA+2")
        .astype("int8")
    )

    print()
    print("Target:")
    print(
        df["target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return df


# ======================================================================
# SPLIT TEMPORAL
# ======================================================================

def construir_split(
    df: pd.DataFrame,
    seed: int,
):

    """
    Split simultaneamente:
      - out-of-time: TRAIN 202104 / TEST 202105
      - out-of-client: ningun cliente aparece en ambos conjuntos

    Los BAJA+2 ya son naturalmente disjuntos entre ambos meses.
    Los clientes FIEL se particionan deterministicamente segun seed.

    Se conserva aproximadamente la misma prevalencia BAJA+2
    en TRAIN y TEST.
    """

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

    ids_fiel = sorted(
        ids_fiel_train
    )

    if len(ids_fiel) != 4067:
        raise ValueError(
            f"Se esperaban 4067 clientes FIEL; "
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

    ids_barajados = np.array(
        ids_fiel,
        dtype=object,
    )

    rng.shuffle(
        ids_barajados
    )

    ids_train = set(
        ids_barajados[
            :n_fiel_train
        ].tolist()
    )

    ids_test = set(
        ids_barajados[
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

    train = (
        pd.concat(
            [
                baja_train,
                fiel_train,
            ],
            ignore_index=True,
        )
        .reset_index(drop=True)
    )

    test = (
        pd.concat(
            [
                baja_test,
                fiel_test,
            ],
            ignore_index=True,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Controles estrictos
    # --------------------------------------------------------

    clientes_train = set(
        train["numero_de_cliente"]
    )

    clientes_test = set(
        test["numero_de_cliente"]
    )

    inter_total = (
        clientes_train
        & clientes_test
    )

    baja_ids_train = set(
        baja_train["numero_de_cliente"]
    )

    baja_ids_test = set(
        baja_test["numero_de_cliente"]
    )

    inter_baja = (
        baja_ids_train
        & baja_ids_test
    )

    inter_fiel = (
        ids_train
        & ids_test
    )

    if inter_total:
        raise ValueError(
            f"Leakage de clientes: "
            f"{len(inter_total)} clientes aparecen "
            f"en TRAIN y TEST."
        )

    if inter_baja:
        raise ValueError(
            f"Hay {len(inter_baja)} BAJA+2 "
            f"repetidos entre TRAIN y TEST."
        )

    if inter_fiel:
        raise ValueError(
            f"Hay {len(inter_fiel)} FIEL "
            f"repetidos entre TRAIN y TEST."
        )

    if len(baja_train) != 865:
        raise ValueError(
            f"BAJA+2 TRAIN esperado=865; "
            f"obtenido={len(baja_train)}."
        )

    if len(baja_test) != 1097:
        raise ValueError(
            f"BAJA+2 TEST esperado=1097; "
            f"obtenido={len(baja_test)}."
        )

    if len(fiel_train) != 1793:
        raise ValueError(
            f"FIEL TRAIN esperado=1793; "
            f"obtenido={len(fiel_train)}."
        )

    if len(fiel_test) != 2274:
        raise ValueError(
            f"FIEL TEST esperado=2274; "
            f"obtenido={len(fiel_test)}."
        )

    if train["target"].nunique() != 2:
        raise ValueError(
            "TRAIN no tiene ambas clases."
        )

    if test["target"].nunique() != 2:
        raise ValueError(
            "TEST no tiene ambas clases."
        )

    diagnostico = {
        "seed": seed,
        "n_train": len(train),
        "n_test": len(test),
        "baja_train": len(baja_train),
        "fiel_train": len(fiel_train),
        "baja_test": len(baja_test),
        "fiel_test": len(fiel_test),
        "clientes_train":
            len(clientes_train),
        "clientes_test":
            len(clientes_test),
        "interseccion_clientes":
            len(inter_total),
        "interseccion_baja":
            len(inter_baja),
        "interseccion_fiel":
            len(inter_fiel),
        "prevalencia_train":
            float(train["target"].mean()),
        "prevalencia_test":
            float(test["target"].mean()),
    }

    return train, test, diagnostico


# ======================================================================
# FAMILIAS DISPONIBLES
# ======================================================================

def detectar_familias(
    train: pd.DataFrame,
):

    titulo("DETECCION DE FEATURES")

    todas = list(train.columns)

    t0 = detectar_columnas(
        todas,
        "__t0",
    )

    media2 = detectar_columnas(
        todas,
        "__media2",
    )

    t0_validas = columnas_validas_train(
        train,
        t0,
    )

    media2_validas = columnas_validas_train(
        train,
        media2,
    )

    print(
        f"t0 detectadas:       {len(t0)}"
    )

    print(
        f"t0 validas TRAIN:    "
        f"{len(t0_validas)}"
    )

    print(
        f"media2 detectadas:   "
        f"{len(media2)}"
    )

    print(
        f"media2 validas TRAIN:"
        f" {len(media2_validas)}"
    )

    return (
        t0_validas,
        media2_validas,
    )


# ======================================================================
# RANKING GENERICO POR GAIN - SOLO TRAIN
# ======================================================================

def ranking_gain_train(
    train: pd.DataFrame,
    columnas_modelo: list[str],
    columnas_rankear: list[str],
    nombre: str,
):

    titulo(
        f"RANKING {nombre} — "
        "EXCLUSIVAMENTE TRAIN"
    )

    columnas_modelo = list(
        dict.fromkeys(
            columnas_modelo
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
                "feature":
                    columnas_modelo,
                "gain": gain,
                "gain_share": share,
            }
        )

        tmp = tmp[
            tmp["feature"]
            .isin(permitidas)
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

    ranking_seed.to_csv(
        OUT_DIR
        / f"ranking_{nombre}_por_seed.csv",
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

    resumen = resumen.sort_values(
        [
            "gain_share_mean",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    resumen["rank"] = (
        np.arange(
            1,
            len(resumen) + 1,
        )
    )

    resumen.to_csv(
        OUT_DIR
        / f"ranking_{nombre}.csv",
        index=False,
    )

    print()
    print(
        resumen.head(20)
        .to_string(index=False)
    )

    return resumen


# ======================================================================
# RECONSTRUCCION EXACTA DEL TOP40 MEDIA2
# ======================================================================

def reconstruir_media2_top40(
    train: pd.DataFrame,
    t0_cols: list[str],
    media2_cols: list[str],
):

    titulo(
        "RECONSTRUCCION DE C0 — "
        "TOP40 MEDIA2"
    )

    columnas_modelo = (
        list(t0_cols)
        + list(media2_cols)
    )

    ranking = ranking_gain_train(
        train=train,
        columnas_modelo=columnas_modelo,
        columnas_rankear=media2_cols,
        nombre="media2_train",
    )

    permitidas = set(
        media2_cols
    )

    orden = [
        c
        for c in ranking[
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
            "40 media2."
        )

    pd.DataFrame(
        {
            "rank":
                np.arange(
                    1,
                    len(top40) + 1,
                ),
            "feature":
                top40,
        }
    ).to_csv(
        OUT_DIR
        / "media2_top40.csv",
        index=False,
    )

    print()
    print(
        f"Top40 media2: "
        f"{len(top40)}"
    )

    return top40


# ======================================================================
# CONSTRUCCION DE C0
# ======================================================================

def construir_c0(
    t0_cols: list[str],
    media2_top40: list[str],
):

    titulo("CONSTRUCCION DE C0")

    c0 = list(
        dict.fromkeys(
            list(t0_cols)
            + list(media2_top40)
        )
    )

    n_t0 = sum(
        c.endswith("__t0")
        for c in c0
    )

    n_media = sum(
        c.endswith("__media2")
        for c in c0
    )

    print(
        f"C0 total:  {len(c0)}"
    )

    print(
        f"  t0:      {n_t0}"
    )

    print(
        f"  media2:  {n_media}"
    )

    pd.DataFrame(
        {
            "feature": c0,
        }
    ).to_csv(
        OUT_DIR
        / "features_c0.csv",
        index=False,
    )

    return c0


# ======================================================================
# RANKING CONJUNTO DE C0
# ======================================================================

def ranking_conjunto_c0(
    train: pd.DataFrame,
    c0: list[str],
):

    return ranking_gain_train(
        train=train,
        columnas_modelo=c0,
        columnas_rankear=c0,
        nombre="c0_conjunto",
    )


# ======================================================================
# DISENOS D0-D6
# ======================================================================

def construir_disenos(
    c0: list[str],
    ranking_c0: pd.DataFrame,
):

    titulo(
        "DISEÑOS CONGELADOS Z513"
    )

    permitidas = set(c0)

    orden = [
        c
        for c in ranking_c0["feature"]
        .astype(str)
        .tolist()
        if c in permitidas
    ]

    if len(c0) != 190:
        raise ValueError(
            f"C0 deberia tener 190 features; "
            f"tiene {len(c0)}."
        )

    if len(orden) != 190:
        raise ValueError(
            f"Ranking C0 deberia tener 190 features; "
            f"tiene {len(orden)}."
        )

    disenos = {
        "F0_c0_top70":
            orden[:70],
        "F1_c0_top100":
            orden[:100],
        "F2_c0_all190":
            list(c0),
    }

    for nombre, cols in disenos.items():

        n_t0 = sum(
            c.endswith("__t0")
            for c in cols
        )

        n_media = sum(
            c.endswith("__media2")
            for c in cols
        )

        print(
            f"{nombre:16s} "
            f"features={len(cols):3d} "
            f"t0={n_t0:3d} "
            f"media2={n_media:3d}"
        )

    filas = []

    for nombre, cols in disenos.items():

        for posicion, col in enumerate(
            cols,
            start=1,
        ):

            filas.append(
                {
                    "experimento":
                        nombre,
                    "posicion":
                        posicion,
                    "feature":
                        col,
                }
            )

    pd.DataFrame(
        filas
    ).to_csv(
        OUT_DIR
        / "features_por_experimento.csv",
        index=False,
    )

    return disenos


# ======================================================================
# EVALUACION
# ======================================================================

def evaluar_disenos(
    df: pd.DataFrame,
    disenos: dict[str, list[str]],
):

    titulo(
        "EVALUACION OUT-OF-TIME + OUT-OF-CLIENT"
    )

    metricas = []
    predicciones = []
    importancias = []
    diagnosticos = []

    # La semilla define simultaneamente:
    # 1) la particion de clientes FIEL
    # 2) la semilla del modelo
    for seed in SEEDS:

        subtitulo(
            f"SPLIT / SEED {seed}"
        )

        (
            train,
            test,
            diagnostico,
        ) = construir_split(
            df=df,
            seed=seed,
        )

        diagnosticos.append(
            diagnostico
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
            "Interseccion clientes TRAIN/TEST: "
            f"{diagnostico['interseccion_clientes']}"
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

            model = nuevo_modelo(
                seed
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

            auc = roc_auc_score(
                y_test,
                prob,
            )

            ap = average_precision_score(
                y_test,
                prob,
            )

            ll = log_loss(
                y_test,
                prob,
                labels=[0, 1],
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
                    "experimento":
                        nombre,
                    "seed":
                        seed,
                    "n_features":
                        len(cols),
                    "n_train":
                        len(train),
                    "n_test":
                        len(test),
                    "auc_test":
                        auc,
                    "ap_test":
                        ap,
                    "logloss_test":
                        ll,
                    "fit_seconds":
                        segundos,
                }
            )

            tmp_pred = pd.DataFrame(
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
                        test[
                            "foto_ancla"
                        ].to_numpy(),
                    "y":
                        y_test,
                    "prob":
                        prob,
                }
            )

            predicciones.append(
                tmp_pred
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

            tmp_imp = pd.DataFrame(
                {
                    "experimento":
                        nombre,
                    "seed":
                        seed,
                    "feature":
                        cols,
                    "gain":
                        gain,
                    "gain_share":
                        share,
                }
            )

            importancias.append(
                tmp_imp
            )

    diagnosticos = pd.DataFrame(
        diagnosticos
    )

    diagnosticos.to_csv(
        OUT_DIR
        / "diagnostico_splits.csv",
        index=False,
    )

    return (
        pd.DataFrame(metricas),
        pd.concat(
            predicciones,
            ignore_index=True,
        ),
        pd.concat(
            importancias,
            ignore_index=True,
        ),
        diagnosticos,
    )


# ======================================================================
# RESUMEN DE MODELOS
# ======================================================================

def resumir_modelos(
    metricas: pd.DataFrame,
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

    resumen = (
        resumen
        .sort_values(
            "n_features"
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


# ======================================================================
# COMPARACION POR SEMILLA CONTRA D6
# ======================================================================

def comparar_con_all(
    metricas: pd.DataFrame,
):

    titulo(
        "COMPARACION POR SEMILLA "
        "CONTRA D6"
    )

    base = (
        metricas[
            metricas["experimento"]
            == "E9_c0_all"
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
                    "auc_all",
                "ap_test":
                    "ap_all",
                "logloss_test":
                    "logloss_all",
            }
        )
    )

    comp = metricas.merge(
        base,
        on="seed",
        how="left",
    )

    comp[
        "delta_auc_vs_all"
    ] = (
        comp["auc_test"]
        - comp["auc_all"]
    )

    comp[
        "delta_ap_vs_all"
    ] = (
        comp["ap_test"]
        - comp["ap_all"]
    )

    comp[
        "delta_logloss_vs_all"
    ] = (
        comp["logloss_test"]
        - comp["logloss_all"]
    )

    comp[
        "mejor_auc_que_all"
    ] = (
        comp["delta_auc_vs_all"]
        > 0
    )

    tabla = (
        comp
        .groupby(
            "experimento",
            as_index=False,
        )
        .agg(
            semillas_mejor_auc=(
                "mejor_auc_que_all",
                "sum",
            ),
            delta_auc_mean=(
                "delta_auc_vs_all",
                "mean",
            ),
            delta_ap_mean=(
                "delta_ap_vs_all",
                "mean",
            ),
            delta_logloss_mean=(
                "delta_logloss_vs_all",
                "mean",
            ),
        )
    )

    print(
        tabla.to_string(
            index=False
        )
    )

    return comp, tabla


# ======================================================================
# ENSEMBLE DE SEMILLAS
# ======================================================================

def ensemble_semillas(
    predicciones: pd.DataFrame,
):

    titulo("ENSEMBLE ENTRE SEMILLAS")

    ens = (
        predicciones
        .groupby(
            [
                "experimento",
                "fila_test",
            ],
            as_index=False,
        )
        .agg(
            y=("y", "first"),
            prob=("prob", "mean"),
        )
    )

    filas = []

    for nombre, g in ens.groupby(
        "experimento"
    ):

        auc = roc_auc_score(
            g["y"],
            g["prob"],
        )

        ap = average_precision_score(
            g["y"],
            g["prob"],
        )

        ll = log_loss(
            g["y"],
            g["prob"],
            labels=[0, 1],
        )

        filas.append(
            {
                "experimento":
                    nombre,
                "n":
                    len(g),
                "auc_ensemble":
                    auc,
                "ap_ensemble":
                    ap,
                "logloss_ensemble":
                    ll,
            }
        )

    resumen = pd.DataFrame(
        filas
    )

    mapa_n = (
        predicciones[
            [
                "experimento",
            ]
        ]
        .drop_duplicates()
    )

    del mapa_n

    resumen = resumen.sort_values(
        "experimento"
    ).reset_index(
        drop=True
    )

    print(
        resumen.to_string(
            index=False
        )
    )

    return ens, resumen


# ======================================================================
# RESUMEN DE IMPORTANCIAS DE TEST MODELS
# ======================================================================

def resumir_importancias(
    importancias: pd.DataFrame,
):

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

    resumen["rank"] = (
        resumen
        .groupby(
            "experimento"
        )["gain_share_mean"]
        .rank(
            method="first",
            ascending=False,
        )
    )

    return resumen


# ======================================================================
# CONTROL CONTRA Z510
# ======================================================================

def validar_d6_con_z510(
    metricas: pd.DataFrame,
):

    titulo(
        "VALIDACION E9 CONTRA C0 DE Z510"
    )

    archivo = (
        Z510_DIR
        / "metricas_corridas.csv"
    )

    if not archivo.exists():

        print(
            "No existe metricas_corridas.csv "
            "de Z510."
        )

        print(
            "Se omite control automatico."
        )

        return None

    z510 = pd.read_csv(
        archivo
    )

    nombre_c0 = (
        "C0_t0_media2_top40"
    )

    ref = z510[
        z510["experimento"]
        == nombre_c0
    ].copy()

    actual = metricas[
        metricas["experimento"]
        == "E9_c0_all"
    ].copy()

    if ref.empty:

        print(
            "No se encontro C0 en Z510."
        )

        return None

    comp = actual[
        [
            "seed",
            "auc_test",
            "ap_test",
            "logloss_test",
        ]
    ].merge(
        ref[
            [
                "seed",
                "auc_test",
                "ap_test",
                "logloss_test",
            ]
        ],
        on="seed",
        suffixes=(
            "_d6",
            "_c0",
        ),
    )

    comp[
        "delta_auc"
    ] = (
        comp["auc_test_d6"]
        - comp["auc_test_c0"]
    )

    comp[
        "delta_ap"
    ] = (
        comp["ap_test_d6"]
        - comp["ap_test_c0"]
    )

    comp[
        "delta_logloss"
    ] = (
        comp["logloss_test_d6"]
        - comp["logloss_test_c0"]
    )

    print(
        comp.to_string(
            index=False
        )
    )

    max_auc = (
        comp["delta_auc"]
        .abs()
        .max()
    )

    print()
    print(
        "Maxima diferencia absoluta "
        f"de AUC: {max_auc:.12f}"
    )

    if max_auc < 1e-12:

        print(
            "CONTROL E9 vs C0: "
            "REPRODUCCION EXACTA."
        )

    else:

        print(
            "ADVERTENCIA: E9 no reproduce "
            "exactamente C0."
        )

    return comp


# ======================================================================
# CONTROL CONTRA Z509 B3
# ======================================================================

def validar_d6_con_z509(
    metricas: pd.DataFrame,
):

    titulo(
        "VALIDACION E9 CONTRA B3 DE Z509"
    )

    archivo = (
        Z509_DIR
        / "metricas_corridas.csv"
    )

    if not archivo.exists():

        print(
            "No existe metricas_corridas.csv "
            "de Z509."
        )

        return None

    z509 = pd.read_csv(
        archivo
    )

    nombre_b3 = (
        "B3_t0_media2_top40"
    )

    ref = z509[
        z509["experimento"]
        == nombre_b3
    ].copy()

    actual = metricas[
        metricas["experimento"]
        == "E9_c0_all"
    ].copy()

    if ref.empty:

        print(
            "No se encontro B3 en Z509."
        )

        return None

    comp = actual[
        [
            "seed",
            "auc_test",
            "ap_test",
            "logloss_test",
        ]
    ].merge(
        ref[
            [
                "seed",
                "auc_test",
                "ap_test",
                "logloss_test",
            ]
        ],
        on="seed",
        suffixes=(
            "_d6",
            "_b3",
        ),
    )

    comp[
        "delta_auc"
    ] = (
        comp["auc_test_d6"]
        - comp["auc_test_b3"]
    )

    comp[
        "delta_ap"
    ] = (
        comp["ap_test_d6"]
        - comp["ap_test_b3"]
    )

    comp[
        "delta_logloss"
    ] = (
        comp["logloss_test_d6"]
        - comp["logloss_test_b3"]
    )

    print(
        comp.to_string(
            index=False
        )
    )

    max_auc = (
        comp["delta_auc"]
        .abs()
        .max()
    )

    print()
    print(
        "Maxima diferencia absoluta "
        f"de AUC: {max_auc:.12f}"
    )

    if max_auc < 1e-12:

        print(
            "CONTROL E9 vs B3: "
            "REPRODUCCION EXACTA."
        )

    else:

        print(
            "ADVERTENCIA: E9 no reproduce "
            "exactamente B3."
        )

    return comp


# ======================================================================
# DIAGNOSTICO DE COMPACTACION
# ======================================================================

def diagnostico_compactacion(
    resumen: pd.DataFrame,
    comparacion_resumen:
        pd.DataFrame,
):

    titulo(
        "DIAGNOSTICO DE COMPACTACION"
    )

    diag = resumen.merge(
        comparacion_resumen[
            [
                "experimento",
                "semillas_mejor_auc",
            ]
        ],
        on="experimento",
        how="left",
    )

    columnas = [
        "experimento",
        "n_features",
        "auc_test_mean",
        "auc_test_std",
        "ap_test_mean",
        "logloss_test_mean",
        "delta_auc_vs_all",
        "delta_ap_vs_all",
        "semillas_mejor_auc",
    ]

    print(
        diag[columnas]
        .to_string(index=False)
    )

    mejor_auc = (
        diag
        .sort_values(
            [
                "auc_test_mean",
                "n_features",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    print()
    print("Mayor AUC medio:")
    print(
        f"  {mejor_auc['experimento']}"
    )

    print(
        f"  AUC = "
        f"{mejor_auc['auc_test_mean']:.6f}"
    )

    print(
        f"  features = "
        f"{int(mejor_auc['n_features'])}"
    )

    return diag


# ======================================================================
# EXPORTACION
# ======================================================================

def exportar(
    metricas,
    resumen,
    comparacion,
    comparacion_resumen,
    predicciones,
    pred_ensemble,
    resumen_ensemble,
    importancias,
    importancia_resumen,
    diagnostico,
    control_z510,
    control_z509,
):

    titulo("EXPORTACION")

    objetos = {
        "metricas_corridas.csv":
            metricas,
        "resumen_modelos.csv":
            resumen,
        "comparacion_all.csv":
            comparacion,
        "comparacion_resumen.csv":
            comparacion_resumen,
        "predicciones_test.csv":
            predicciones,
        "predicciones_ensemble.csv":
            pred_ensemble,
        "resumen_ensemble.csv":
            resumen_ensemble,
        "importancia_gain.csv":
            importancias,
        "importancia_resumen.csv":
            importancia_resumen,
        "diagnostico_compactacion.csv":
            diagnostico,
    }

    if control_z510 is not None:

        objetos[
            "control_d6_vs_z510.csv"
        ] = control_z510

    if control_z509 is not None:

        objetos[
            "control_d6_vs_z509.csv"
        ] = control_z509

    for nombre, df in objetos.items():

        ruta = (
            OUT_DIR
            / nombre
        )

        df.to_csv(
            ruta,
            index=False,
        )

        print(
            f"OK  {nombre:35s} "
            f"{ruta.stat().st_size:>12,} bytes"
        )


# ======================================================================
# RESUMEN TXT
# ======================================================================

def guardar_resumen_txt(
    resumen: pd.DataFrame,
    diagnostico: pd.DataFrame,
    segundos: float,
):

    ruta = (
        OUT_DIR
        / "resumen_z513.txt"
    )

    columnas = [
        "experimento",
        "n_features",
        "auc_test_mean",
        "auc_test_std",
        "ap_test_mean",
        "logloss_test_mean",
        "delta_auc_vs_all",
        "delta_ap_vs_all",
        "semillas_mejor_auc",
    ]

    with open(
        ruta,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Z513 - VALIDACION OUT-OF-TIME + OUT-OF-CLIENT\n"
        )

        f.write(
            "=" * 78
            + "\n\n"
        )

        f.write(
            f"TRAIN: {TRAIN_FOTO}\n"
        )

        f.write(
            f"TEST: {TEST_FOTO}\n"
        )

        f.write(
            "SEEDS: "
            + ", ".join(
                map(str, SEEDS)
            )
            + "\n\n"
        )

        f.write(
            "RESULTADOS\n"
        )

        f.write(
            diagnostico[
                columnas
            ].to_string(
                index=False
            )
        )

        f.write("\n\n")

        mejor = (
            resumen
            .sort_values(
                [
                    "auc_test_mean",
                    "n_features",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

        f.write(
            "Mayor AUC medio:\n"
        )

        f.write(
            f"{mejor['experimento']}\n"
        )

        f.write(
            f"AUC="
            f"{mejor['auc_test_mean']:.6f}\n"
        )

        f.write(
            f"Features="
            f"{int(mejor['n_features'])}\n"
        )

        f.write(
            f"\nTiempo total: "
            f"{segundos:.2f} s\n"
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
        "Z513 — VALIDACION OUT-OF-TIME + OUT-OF-CLIENT"
    )

    print(
        f"TRAIN = {TRAIN_FOTO}"
    )

    print(
        f"TEST  = {TEST_FOTO}"
    )

    print(
        "Seeds = "
        + ", ".join(
            map(str, SEEDS)
        )
    )

    # --------------------------------------------------------------
    # Datos
    # --------------------------------------------------------------

    df = cargar_datos()

    # --------------------------------------------------------------
    # C0 y ranking congelados de Z511
    # --------------------------------------------------------------

    archivo_c0 = (
        Z511_DIR
        / "features_c0.csv"
    )

    archivo_ranking = (
        Z511_DIR
        / "ranking_c0_conjunto.csv"
    )

    if not archivo_c0.exists():
        raise FileNotFoundError(
            f"No existe C0 congelado de Z511: {archivo_c0}"
        )

    if not archivo_ranking.exists():
        raise FileNotFoundError(
            f"No existe ranking congelado de Z511: {archivo_ranking}"
        )

    c0 = (
        pd.read_csv(archivo_c0)["feature"]
        .astype(str)
        .tolist()
    )

    ranking_c0 = pd.read_csv(
        archivo_ranking
    )

    if len(c0) != 190:
        raise ValueError(
            f"C0 congelado deberia tener 190 features; tiene {len(c0)}."
        )

    if c0 != list(dict.fromkeys(c0)):
        raise ValueError(
            "C0 congelado contiene features duplicadas."
        )

    faltantes_dataset = [
        c
        for c in c0
        if c not in df.columns
    ]

    if faltantes_dataset:
        raise ValueError(
            "Hay features de C0 que no existen en el dataset: "
            + ", ".join(faltantes_dataset)
        )

    ranking_features = (
        ranking_c0["feature"]
        .astype(str)
        .tolist()
    )

    if set(ranking_features) != set(c0):
        raise ValueError(
            "El ranking congelado y C0 no contienen "
            "exactamente las mismas features."
        )

    print()
    print(
        f"C0 congelado Z511: {len(c0)} features"
    )

    print(
        f"Ranking congelado Z511: "
        f"{len(ranking_c0)} features"
    )

    # --------------------------------------------------------------
    # Diseños
    # --------------------------------------------------------------

    disenos = construir_disenos(
        c0=c0,
        ranking_c0=ranking_c0,
    )

    # --------------------------------------------------------------
    # Evaluacion out-of-time
    # --------------------------------------------------------------

    (
        metricas,
        predicciones,
        importancias,
        diagnosticos,
    ) = evaluar_disenos(
        df=df,
        disenos=disenos,
    )

    # --------------------------------------------------------------
    # Resumen de las cinco replicas out-of-client
    # --------------------------------------------------------------

    resumen = resumir_modelos(
        metricas
    )

    # --------------------------------------------------------------
    # Importancias
    # --------------------------------------------------------------

    importancia_resumen = (
        resumir_importancias(
            importancias
        )
    )

    # --------------------------------------------------------------
    # Exportar resultados Z513
    # --------------------------------------------------------------

    metricas.to_csv(
        OUT_DIR / "metricas_corridas.csv",
        index=False,
    )

    resumen.to_csv(
        OUT_DIR / "resumen_modelos.csv",
        index=False,
    )

    predicciones.to_csv(
        OUT_DIR / "predicciones_test.csv",
        index=False,
    )

    importancias.to_csv(
        OUT_DIR / "importancia_gain.csv",
        index=False,
    )

    importancia_resumen.to_csv(
        OUT_DIR / "importancia_resumen.csv",
        index=False,
    )

    diagnosticos.to_csv(
        OUT_DIR / "diagnostico_splits.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Control global de leakage
    # --------------------------------------------------------------

    if not (
        diagnosticos[
            "interseccion_clientes"
        ].eq(0).all()
    ):
        raise ValueError(
            "Al menos un split tiene clientes "
            "repetidos entre TRAIN y TEST."
        )

    if not (
        diagnosticos[
            "interseccion_baja"
        ].eq(0).all()
    ):
        raise ValueError(
            "Al menos un split tiene BAJA+2 "
            "repetidos entre TRAIN y TEST."
        )

    if not (
        diagnosticos[
            "interseccion_fiel"
        ].eq(0).all()
    ):
        raise ValueError(
            "Al menos un split tiene FIEL "
            "repetidos entre TRAIN y TEST."
        )

    # --------------------------------------------------------------
    # Comparacion descriptiva contra All190
    # --------------------------------------------------------------

    base = (
        resumen[
            resumen["experimento"]
            == "F2_c0_all190"
        ]
        .iloc[0]
    )

    resumen = resumen.copy()

    resumen["delta_auc_vs_all190"] = (
        resumen["auc_test_mean"]
        - base["auc_test_mean"]
    )

    resumen["delta_ap_vs_all190"] = (
        resumen["ap_test_mean"]
        - base["ap_test_mean"]
    )

    resumen[
        "delta_logloss_vs_all190"
    ] = (
        resumen["logloss_test_mean"]
        - base["logloss_test_mean"]
    )

    resumen.to_csv(
        OUT_DIR / "resumen_modelos.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Resumen TXT
    # --------------------------------------------------------------

    segundos = (
        time.time()
        - inicio_total
    )

    archivo_resumen = (
        OUT_DIR
        / "resumen_z513.txt"
    )

    with open(
        archivo_resumen,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Z513 - VALIDACION OUT-OF-TIME "
            "+ OUT-OF-CLIENT\n"
        )

        f.write(
            "=" * 78
            + "\n\n"
        )

        f.write(
            f"TRAIN temporal: {TRAIN_FOTO}\n"
        )

        f.write(
            f"TEST temporal : {TEST_FOTO}\n"
        )

        f.write(
            "Seeds/splits  : "
            + ", ".join(
                map(str, SEEDS)
            )
            + "\n\n"
        )

        f.write(
            "Cada seed define una particion "
            "disjunta de clientes FIEL "
            "y la semilla del modelo.\n"
        )

        f.write(
            "Las features permanecen "
            "congeladas desde Z511/Z512.\n\n"
        )

        f.write(
            "DIAGNOSTICO DE SPLITS\n"
        )

        f.write(
            diagnosticos.to_string(
                index=False
            )
        )

        f.write(
            "\n\nRESULTADOS\n"
        )

        columnas_txt = [
            "experimento",
            "n_features",
            "auc_test_mean",
            "auc_test_std",
            "ap_test_mean",
            "ap_test_std",
            "logloss_test_mean",
            "delta_auc_vs_all190",
            "delta_ap_vs_all190",
            "delta_logloss_vs_all190",
        ]

        f.write(
            resumen[
                columnas_txt
            ].to_string(
                index=False
            )
        )

        f.write(
            f"\n\nTiempo total: "
            f"{segundos:.2f} s\n"
        )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    titulo(
        "RESULTADO FINAL Z513"
    )

    print()
    print(
        "Diagnostico de splits:"
    )

    columnas_split = [
        "seed",
        "n_train",
        "n_test",
        "baja_train",
        "fiel_train",
        "baja_test",
        "fiel_test",
        "interseccion_clientes",
        "prevalencia_train",
        "prevalencia_test",
    ]

    print(
        diagnosticos[
            columnas_split
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Metricas:"
    )

    columnas_finales = [
        "experimento",
        "n_features",
        "auc_test_mean",
        "auc_test_std",
        "ap_test_mean",
        "ap_test_std",
        "logloss_test_mean",
        "delta_auc_vs_all190",
        "delta_ap_vs_all190",
        "delta_logloss_vs_all190",
    ]

    print(
        resumen[
            columnas_finales
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Control leakage: "
        "0 clientes compartidos en "
        "todos los splits."
    )

    print()
    print(
        f"Tiempo total: "
        f"{segundos:.2f} s"
    )

    print()
    print(
        "Z513 FINALIZADO CORRECTAMENTE."
    )


if __name__ == "__main__":
    main()